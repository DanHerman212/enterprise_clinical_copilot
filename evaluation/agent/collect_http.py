"""Collect eval traces from the DEPLOYED agent over HTTP.

Why this exists rather than reusing `collect.py`: that collector runs the graph
in-process with the MCP toolbox over stdio, which measures the repository's code
and the live endpoints but not the deployment. The deployed service is what
users reach, and it is the thing whose behaviour an evaluation should describe —
a different container, a different concurrency profile, a different revision.

The trade this makes, stated so nobody has to rediscover it: the local
collector can read the agent's internal state directly, while this one has only
the response body. That is sufficient here, and it was verified rather than
assumed — `/ask` returns `tool_calls` with each call's `response` (including the
full retrieved passages for `rag_search` and the predict payload with its
threshold, decision and top factors) and `sources` with the passage text, which
is the evidence the judge scores groundedness against. It also returns
`langfuse_trace_id`, so scores can be attached to the trace the deployment
produced rather than to a trace this script made up.

What it deliberately does NOT do: aggregate scores or judge anything. It
collects, gates its own run against the retrieval-failure rate, and writes JSONL
in the shape `judge.py` already reads.

Usage (harness root):
    # preflight only: six smoke questions, then stop
    .venv/bin/python evaluation/agent/collect_http.py --preflight-only

    # a pilot: three patients x three prompts, plus a first adversarial batch
    .venv/bin/python evaluation/agent/collect_http.py --max-patients 3

    # the full census (82 x 3 = 246 clinical + 20 adversarial), resumable
    .venv/bin/python evaluation/agent/collect_http.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from health import classify_trace, format_summary, retrieval_failed, summarize  # noqa: E402
from prompt_set import (  # noqa: E402
    COMMAND_FROM_WIRE,
    COMMAND_TOOLS,
    COVERAGE_COMMANDS,
    PROMPT_NAMES,
    SLASH_COMMANDS,
    follow_up_for,
    question_for,
)
from services.agent.contracts import (  # noqa: E402
    MAX_REPLAYED_TOOL_CALLS,
    MAX_TOOL_NAME_CHARS,
    MAX_TURN_ANSWER_CHARS,
)

RESULTS = HERE / "results"
# The clinical census: 82 inpatient admissions, built from the served feature
# table by build_census.py. The five outpatient encounters are written
# alongside it but are not part of this set — they have no inpatient risk score
# to grade.
CASES = RESULTS / "cases_82.json"
#: The targeted modality matrix. A separate file from the census because these are
#: single-turn probes chosen to exercise one vector each, not admissions to sweep.
SUITE2 = HERE / "suite2_cases.json"
ADVERSARIAL = HERE / "adversarial_cases.json"
OUT = RESULTS / "traces_http.jsonl"
ADV_OUT = RESULTS / "adversarial_http.jsonl"

# The site's mark for an answer it had to shorten for a replay. The same string
# for the same reason: an answer shortened silently would tell the model that a
# truncated answer is the whole of what the clinician was told.
TRUNCATION_MARK = "\n[earlier answer truncated for replay]"

# A single answer took 7.4s against the deployed service when the container was
# warm, and the timeout has to cover a cold start of both the agent and the MCP
# server, which the platform bounds well below this.
_ASK_TIMEOUT_SECONDS = 240


def _identity_token(audience: str | None = None) -> str:
    """An identity token for the private agent.

    Two sources, because this runs in two places. Inside a Cloud Run job there is no
    gcloud CLI at all — the token comes from the metadata server, for the job's own
    service account, with the service URL as the audience the IAM check compares
    against. On a laptop that call has no metadata server and raises, so the CLI
    remains the fallback; it is also what the site does (`demo/agent_client.py`), which
    keeps one story about how this service is authenticated.
    """
    try:
        from google.auth.transport import requests as google_requests  # noqa: PLC0415
        from google.oauth2 import id_token  # noqa: PLC0415

        if audience:
            return id_token.fetch_id_token(google_requests.Request(), audience)
    except Exception:  # noqa: BLE001 — no metadata server; fall through to the CLI
        pass

    proc = subprocess.run(
        ["gcloud", "auth", "print-identity-token"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(f"could not get an identity token: {proc.stderr.strip()}")
    return proc.stdout.strip()


class AgentClient:
    """POSTs one question to the deployed agent, with token refresh."""

    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip("/")
        self._token = token
        self._lock = asyncio.Lock()
        self._client = httpx.AsyncClient(timeout=_ASK_TIMEOUT_SECONDS)

    async def _refresh(self) -> None:
        async with self._lock:
            # Another task may have refreshed while this one waited.
            self._token = await asyncio.to_thread(_identity_token, self.base_url)

    async def ask(self, body: dict) -> tuple[int, dict]:
        """Return (status_code, json_body). Raises on transport failure.

        The token lives in `self._token` and is read directly rather than
        through a `_token()` accessor: an accessor with that name and an
        attribute with that name are the same name, and the first version of
        this class had both, so every request tried to call a string and failed
        with a bare `TypeError` that said nothing about tokens.
        """
        for attempt in range(2):
            resp = await self._client.post(
                f"{self.base_url}/ask",
                json=body,
                headers={"Authorization": f"Bearer {self._token}"},
            )
            if resp.status_code == 401 and attempt == 0:
                # An identity token lives an hour; a long run can outlive one.
                await self._refresh()
                continue
            try:
                return resp.status_code, resp.json()
            except Exception:
                return resp.status_code, {"_raw": resp.text[:2000]}
        return 401, {"error": "unauthenticated"}

    async def stream(self, body: dict) -> tuple[int, list[tuple[str, dict]]]:
        """POST with `stream_prose` and collect the SSE frames.

        Only the preflight uses this. The batch census stays blocking, because the
        Reconciliation badge is an SSE stage that a blocking caller never receives —
        so the census records `reconciled: None` (meaning "not observable here")
        rather than implying the badge did not fire.
        """
        events: list[tuple[str, dict]] = []
        async with self._client.stream(
            "POST", f"{self.base_url}/ask/stream", json={**body, "stream_prose": True},
            headers={"Authorization": f"Bearer {self._token}",
                     "Accept": "text/event-stream"},
        ) as resp:
            stage = ""
            async for line in resp.aiter_lines():
                if line.startswith("event: "):
                    stage = line[len("event: "):].strip()
                elif line.startswith("data: "):
                    try:
                        events.append((stage, json.loads(line[len("data: "):])))
                    except json.JSONDecodeError:
                        continue
            return resp.status_code, events

    async def aclose(self) -> None:
        await self._client.aclose()


async def _with_retries(client: AgentClient, body: dict, attempts: int = 3):
    """Retry transport failures and 5xx/429; return (status, body) either way.

    A 4xx is NOT retried: the contract refusing a request is an answer, and it is
    the answer the boundary cases are asking for. Retrying it would turn a
    correct refusal into a latency problem.
    """
    last_exc = None
    for i in range(attempts):
        try:
            status, payload = await client.ask(body)
            if status in (429, 500, 502, 503, 504) and i < attempts - 1:
                await asyncio.sleep(2 + 3 * i)
                continue
            return status, payload
        except Exception as exc:  # transport-level
            last_exc = exc
            if i < attempts - 1:
                await asyncio.sleep(2 + 3 * i)
    raise last_exc if last_exc else RuntimeError("no response")


def _clamp(answer: str) -> str:
    """An answer short enough to be replayed, marked when it was shortened."""
    if len(answer) <= MAX_TURN_ANSWER_CHARS:
        return answer
    return answer[: MAX_TURN_ANSWER_CHARS - len(TRUNCATION_MARK)] + TRUNCATION_MARK


def replay_turn(asked: str, record: dict) -> dict:
    """One answered turn, in the shape the service replays.

    The agent holds nothing between requests, so a second turn about the first
    answer is answerable only if the caller sends the first turn back with it.
    The shape is the contract's: `turns` is a list of `{question, answer,
    tool_calls}`, at most six of them, and no other conversation field is
    accepted. The site builds the same structure from its own store.

    A replayed tool call carries `name`, `args` and a `payload` only where the
    result cannot be obtained again. Retrieval returns discharge-note text and is
    resolved again from the corpus, so its payload is left out; a prediction is
    kept, because re-deriving it re-bills an endpoint and the score behind a
    published answer should be the score that was shown.

    Built from the record this collector wrote, so the turn the service replays
    is the turn it answered and not a paraphrase of it.
    """
    calls = []
    for call in (record.get("tool_calls") or [])[:MAX_REPLAYED_TOOL_CALLS]:
        if not isinstance(call, dict):
            continue
        name = call.get("name")
        if not isinstance(name, str) or not name or len(name) > MAX_TOOL_NAME_CHARS:
            continue
        entry = {
            "name": name,
            "args": call.get("args") if isinstance(call.get("args"), dict) else {},
        }
        if not call.get("derivable", True):
            payload = call.get("response")
            if payload is not None:
                entry["payload"] = payload
        calls.append(entry)
    return {
        "question": asked,
        "answer": _clamp(record.get("answer") or ""),
        "tool_calls": calls,
    }


def _components(node, out: set[str] | None = None) -> list[str]:
    """Every A2UI component name in an envelope, however deeply nested.

    The envelope is A2UI messages, not a flat list: the components live under
    `updateComponents.components`, and the root is a `Column` whose children carry
    the card. Walking generically means a surface that gains a wrapper does not
    silently report zero components to an assertion that is looking for one.
    """
    if out is None:
        out = set()
    if isinstance(node, dict):
        name = node.get("component")
        if isinstance(name, str):
            out.add(name)
        for value in node.values():
            _components(value, out)
    elif isinstance(node, list):
        for item in node:
            _components(item, out)
    return sorted(out)


def _record_from_response(case: dict, status: int, payload: dict,
                          turn: int = 1, question: str | None = None,
                          reconciled: str | None = None) -> dict:
    """One trace record, in the shape judge.py reads.

    Field names match `collect.py` exactly — `hadm_id`, `prompt`, `question`,
    `answer`, `tool_calls` — because the judge and the archive readers key off
    them, and a second collector inventing its own schema would silently produce
    traces nothing downstream could read.
    """
    rec = {
        "hadm_id": case.get("hadm_id"),
        # The question actually put, which for a second turn is the follow-up and
        # not the case's opening question.
        "question": question if question is not None else case["question"],
        "turn": turn,
        "at": datetime.now(timezone.utc).isoformat(),
        "transport": "http",
        "status": status,
    }
    # Three case shapes reach here, and `id` alone does not identify which: a Suite 2
    # modality case carries an `id` and no `family`. Keying the adversarial branch on
    # `"id" in case` raised KeyError: 'family' on the pilot's first Suite 2 request and
    # took the whole run down, so each shape is named by the field only it has.
    if "family" in case:  # adversarial probe
        rec.update({"id": case["id"], "family": case["family"],
                    "prompt": "adversarial", "expect": case["expect"]})
    elif "id" in case:  # Suite 2 targeted modality case
        rec.update({"id": case["id"], "prompt": case.get("prompt") or "suite2",
                    "expect": case.get("expect") or {}})
    else:  # census case
        rec.update({"prompt": case["prompt"], "probability": case.get("probability"),
                    "band": case.get("band"),
                    "readmission_30d": case.get("readmission_30d")})

    if status != 200:
        # A refusal by the contract. Recorded with the code and message the
        # service returned, which are part of the published contract, and no
        # more: an error body is not somewhere to paste a stack trace.
        rec["contract_refusal"] = True
        rec["error"] = str(payload.get("error") or status)
        rec["error_message"] = str(payload.get("message") or "")[:300]
        return rec

    rec["answer"] = payload.get("answer") or ""
    rec["tool_calls"] = payload.get("tool_calls") or []
    rec["sources"] = payload.get("sources") or []
    rec["guardrail_flags"] = payload.get("guardrail_flags") or []
    rec["model"] = payload.get("model")
    rec["code_revision"] = payload.get("code_revision")
    # Structural telemetry, read from the envelope rather than inferred from the
    # prose. Everything downstream — coverage, routing, component and fast-fail
    # assertions — runs on these fields, so a command that fell through to the
    # reasoning loop is visible as data instead of looking like a healthy turn.
    wire_command = payload.get("command")
    # Reported in the harness's own vocabulary. `/ehr` crosses the wire as
    # `summarize`, and an assertion comparing "ehr" to "summarize" would report a
    # command that ran perfectly well as missing.
    rec["command_wire"] = wire_command
    rec["command_executed"] = COMMAND_FROM_WIRE.get(wire_command, wire_command)
    # `routed_to` arrives un-slashed ("labs"); the case files and the product's own
    # routing pill both speak with one ("/labs"), so it is normalised once here.
    routed = payload.get("routed_to")
    rec["routed_to"] = f"/{routed.lstrip('/')}" if routed else None
    rec["a2ui_components"] = _components(payload.get("a2ui"))
    # Literature records the canvas draws, counted from the payload rather than from
    # the prose: a PMID citation is a record in this list, not a marker in the answer.
    lit = payload.get("literature")
    if isinstance(lit, dict):
        lit = lit.get("articles") or lit.get("results") or []
    rec["literature_count"] = len(lit) if isinstance(lit, list) else 0
    # The Reconciled badge is the SSE stage `prose_settled`, which only a caller that
    # asked to watch prose arrive ever receives. A blocking collection therefore
    # records None, and None here means "not observable on this path", not "nothing
    # moved" — the distinction the assertion reads.
    rec["reconciled"] = reconciled
    if case.get("run_tag"):
        # Client-side grouping so Turn 1 and Turn 2 of one trajectory are
        # inspectable together. The agent holds no session state, so this is the
        # harness's own key and travels only in trace metadata.
        rec["session"] = (
            f"{case['run_tag']}:{rec.get('hadm_id')}:{rec.get('prompt')}"
        )
    if payload.get("langfuse_trace_id"):
        rec["langfuse_trace_id"] = payload["langfuse_trace_id"]
    return rec


def _load_done(path: Path, key_fields: tuple[str, ...]) -> dict[tuple, dict]:
    """The records already written, by key, so a restart continues.

    A record whose transport failed is deliberately left out: the question was
    never put to the agent, and treating it as answered is how a re-run reports a
    verdict on a case the service never saw.

    The records themselves are returned rather than only their keys, because a
    half-finished case has to be resumed from its first answer: the second turn
    cannot be built without it, and posting the first turn again would leave two
    records for one answer.
    """
    done: dict[tuple, dict] = {}
    if not path.exists():
        return done
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("transport_error"):
            continue
        done[tuple(rec.get(f) for f in key_fields)] = rec
    return done


def _body(question: str, case: dict) -> dict:
    """The request body for one question."""
    body = {"question": question}
    if case.get("hadm_id"):
        body["hadm_id"] = case["hadm_id"]
    # The run label travels with every request, so the trace this turn produces can
    # be found by tag afterwards instead of by guessing when the run happened.
    if case.get("run_tag"):
        body["run_tag"] = case["run_tag"]
    return body


def _turn2_body(case: dict, replayed: dict) -> dict:
    """The follow-up request body: Turn 1's fields plus the replayed turn.

    Built from `_body` so every case-scoped field survives — `run_tag` above all. A
    literal dict here is what orphaned every follow-up trace in Langfuse: Turn 1 was
    tagged `run:<tag>` and Turn 2 was not, so filtering by run showed half a
    conversation (audited 2026-10-19).
    """
    body = _body(case["follow_up"], case)
    body["turns"] = [replayed]
    return body


def _key(case: dict, turn: int) -> tuple:
    """The key one answer of a case is recorded under, for resume."""
    if "id" in case:
        return (case["id"],)
    return (case.get("hadm_id"), case.get("prompt"), turn)


async def _answer(client, case: dict, turn: int, question: str, body: dict,
                  prior: dict | None = None) -> dict:
    """Post one question and return its record, transport failures included.

    A transport failure is recorded rather than raised: the case is left
    incomplete so a re-run retries it, and the reason survives in the file.
    """
    try:
        status, payload = await _with_retries(client, body)
        rec = _record_from_response(case, status, payload, turn, question)
    except Exception as exc:
        rec = _record_from_response(
            case, 0, {"error": f"{type(exc).__name__}: {exc}"}, turn, question)
        rec["transport_error"] = True
    if prior is not None:
        # What the service was given as the earlier turn. Carried because the
        # conversation exists only in the request the caller sends, so an answer
        # with no record of what preceded it cannot be judged for whether it
        # used the context it was given.
        rec["prior"] = prior
    return rec


def _reachability(recs: list[dict]) -> dict:
    """Failures the retrieval metric cannot see.

    A request that never reached the service, or that came back non-200, has no
    tool calls — so `classify_trace` finds no `missing_text` and no
    `zero_passage`, and a gate that counts only retrieval failures scores it as
    perfectly healthy. The first version of this file did exactly that: all six
    preflight questions failed at the transport layer and the preflight printed
    PASS, because the gate was measuring the absence of evidence as success.

    Kept separate from `summarize()` rather than folded into it, because the two
    mean opposite things: a retrieval failure says the service answered from the
    wrong corpus, a reachability failure says the service did not answer at all.
    """
    transport = sum(1 for r in recs if r.get("transport_error"))
    non_200 = sum(1 for r in recs
                  if not r.get("transport_error") and r.get("status") != 200)
    n = len(recs) or 1
    return {"n": len(recs), "transport_error": transport, "non_200": non_200,
            "unusable": transport + non_200,
            "unusable_rate": (transport + non_200) / n}


def _suite2_cases(args) -> list[dict]:
    """The targeted modality cases, in the same shape the collector runs.

    One turn each, and no follow-up: each case exists to assert one thing about one
    vector — that a click is not routed, that a typo is refused, that an outpatient
    note abstains — and a second turn would add a referent the assertion does not
    depend on.
    """
    if not SUITE2.exists():
        return []
    payload = json.loads(SUITE2.read_text())
    cases: list[dict] = []
    for spec in payload.get("cases") or []:
        case = {
            "id": spec["id"],
            "hadm_id": spec.get("hadm_id"),
            "prompt": "suite2",
            "question": spec["question"],
            "expect": spec.get("expect") or {},
        }
        if args.run_tag:
            case["run_tag"] = args.run_tag
        cases.append(case)
    return cases


def _build_cases(args) -> tuple[list[dict], list[dict]]:
    """(clinical cases, adversarial cases) for this invocation."""
    payload = json.loads(CASES.read_text())
    patients = payload["patients"][: args.max_patients or None]
    prompts = PROMPT_NAMES if args.prompt == "all" else (args.prompt,)

    clinical = []
    # The two phases run as separate jobs, so each one has to be able to ask for its
    # own half. Reading the census file still happens above either way: the file is
    # built from the served table, and a job that could not read it would be a job
    # running against a census nobody can name.
    for p in [] if args.adversarial_only else patients:
        for prompt in prompts:
            clinical.append({
                "hadm_id": p["hadm_id"],
                "prompt": prompt,
                "question": question_for(prompt, p["hadm_id"]),
                # The second turn of the case. Referential: it has no subject
                # without the first answer, which is the property under test.
                "follow_up": follow_up_for(prompt),
                "probability": p.get("probability"),
                "band": p.get("band"),
                "readmission_30d": p.get("readmission_30d"),
                "run_tag": args.run_tag,
            })

    adversarial = []
    if not args.no_adversarial:
        adversarial = json.loads(ADVERSARIAL.read_text())["cases"]
        if args.max_adversarial is not None:
            adversarial = adversarial[: args.max_adversarial]
        adversarial = [{**c, "run_tag": args.run_tag} for c in adversarial]
    return clinical, adversarial


async def _run_set(client, cases, out_path: Path, args, label: str,
                  gate_retrieval: bool = True) -> int:
    """Run one set of cases concurrently, appending to out_path.

    `gate_retrieval` is False for the adversarial set, and the reason is a
    correctness one rather than a convenience: a probe whose correct answer is a
    refusal frequently retrieves nothing, so a zero-passage result there is the
    expected behaviour, not a broken corpus. The reachability check still
    applies to both — a request that never arrived is a bad run either way.
    """
    key_fields = (("id",) if cases and "id" in cases[0]
                  else ("hadm_id", "prompt", "turn"))
    done = _load_done(out_path, key_fields)

    def _complete(case: dict) -> bool:
        """Whether every answer of this case is already in the file."""
        turns = (1, 2) if case.get("follow_up") else (1,)
        return all(_key(case, turn) in done for turn in turns)

    todo = [c for c in cases if not _complete(c)]

    # Counted in answers. A two-turn case is two requests, and a run that stopped
    # after the first turn has done half of a case.
    answers_per_case = 2 if any(c.get("follow_up") for c in cases) else 1
    total = len(cases) * answers_per_case
    print(f"[{label}] {len(cases)} cases ({total} requests), "
          f"{len(done)} already traced, {len(todo)} cases to go", flush=True)
    if not todo:
        return 0

    sem = asyncio.Semaphore(args.concurrency)
    completed = len(done)
    aborted = asyncio.Event()
    lock = asyncio.Lock()
    records: list[dict] = []

    with out_path.open("a") as fh:
        async def one(case: dict) -> None:
            nonlocal completed
            if aborted.is_set():
                return
            async with sem:
                if aborted.is_set():
                    return
                produced: list[dict] = []

                # The first turn. Taken from the file when an earlier run
                # answered it and stopped before the follow-up, so the case is
                # resumed rather than restarted.
                first = done.get(_key(case, 1)) if "id" not in case else None
                if first is None:
                    first = await _answer(client, case, 1, case["question"],
                                          _body(case["question"], case))
                    produced.append(first)

                # The second turn, which carries the first answer back. Skipped
                # when the first turn produced no answer: there is nothing to
                # replay, and a follow-up posted without its referent would be a
                # different question.
                if case.get("follow_up") and first.get("status") == 200:
                    if _key(case, 2) not in done:
                        replayed = replay_turn(case["question"], first)
                        turn2 = _turn2_body(case, replayed)
                        produced.append(await _answer(
                            client, case, 2, case["follow_up"], turn2,
                            prior=replayed))

                if not produced:
                    return
                async with lock:
                    for rec in produced:
                        fh.write(json.dumps(rec, default=str) + "\n")
                    fh.flush()
                    records.extend(produced)
                    completed += len(produced)
                    for rec in produced:
                        status_word = (
                            "ok" if rec.get("status") == 200
                            else f"{rec.get('status')}"
                            if not rec.get("transport_error") else "TRANSPORT")
                        if rec.get("error"):
                            status_word += f" ({rec['error']})"
                        print(f"[{label} {completed}/{total}] "
                              f"{rec.get('hadm_id') or rec.get('id')}/"
                              f"{rec.get('prompt')} t{rec.get('turn')}: "
                              f"{status_word}", flush=True)

                    # Checkpoint the run against the same threshold the
                    # preflight uses. The local runner does this from outside
                    # with a poller because its collector is a separate
                    # process; here the collector is one process, so the check
                    # belongs in the loop that has the data.
                    if (len(records) >= args.check_every
                            and len(records) % args.check_every == 0):
                        s = summarize(records)
                        r = _reachability(records)
                        print("  " + format_summary(s, f"{label} checkpoint"),
                              flush=True)
                        print(f"  {label} reachability: "
                              f"transport_error={r['transport_error']} "
                              f"non_200={r['non_200']}", flush=True)
                        if r["unusable_rate"] > args.max_fail_rate:
                            print(f"[{label}] ABORT: {r['unusable']} of "
                                  f"{r['n']} requests did not produce an "
                                  f"answer — the deployment is not reachable, "
                                  f"and retrieval health cannot be read from "
                                  f"this run.", flush=True)
                            aborted.set()
                            return
                        if gate_retrieval and s["retrieval_fail_rate"] > args.max_fail_rate:
                            print(f"[{label}] ABORT: retrieval-failure rate "
                                  f"{s['retrieval_fail_rate']:.0%} exceeds "
                                  f"{args.max_fail_rate:.0%} — the serving "
                                  f"config is wrong, not the agent.", flush=True)
                            aborted.set()

        await asyncio.gather(*(one(c) for c in todo))

    if aborted.is_set():
        print(f"[{label}] stopped early; {completed}/{total} traced", flush=True)
        return 1
    return 0


async def _route_coverage(client, hadm_id: int) -> int:
    """Assert every covered command actually runs, before the batch is allowed.

    The check the last run did not have. Retrieval health says the deployment can read
    the corpus; it says nothing about whether a command was invoked at all, and a
    command that silently did not run is indistinguishable from a healthy free-text
    turn. `/ehr` ran zero times across 246 turns while every preflight gate reported
    green, because the harness sent the command's prose instead of its slash form and
    prose has no route to `/ehr`.

    One patient, every covered command, and the tool each one must reach.
    """
    print(
        "PREFLIGHT: route coverage — one patient, every covered command …",
        flush=True,
    )
    rc = 0
    for prompt in COVERAGE_COMMANDS:
        # Composed rather than looked up: `/pubmed` is not a question until it has a
        # query, and a bare `/pubmed` would fail for the wrong reason.
        question = question_for(prompt, hadm_id)
        case = {"hadm_id": hadm_id, "prompt": prompt, "question": question}
        rec = await _answer(client, case, 1, question, _body(question, case))
        tools = {c.get("name") for c in rec.get("tool_calls") or []}
        expected = COMMAND_TOOLS[prompt]
        # Two conditions, because either alone is a false pass. The tool says work
        # happened; `command` says the product agrees *which* command ran. A turn that
        # reached the right tool through the reasoning loop has the first and not the
        # second, and that is precisely the failure this gate exists to catch.
        command_ok = rec.get("command_executed") == prompt
        ok = rec.get("status") == 200 and bool(tools & expected) and command_ok
        print(
            f"  {SLASH_COMMANDS[prompt]:9} status={rec.get('status')} "
            f"command={rec.get('command_executed')!r} tools={sorted(tools)} "
            + ("OK" if ok else f"MISSING {sorted(expected) if not tools & expected else 'command-match'}"),
            flush=True,
        )
        if not ok:
            rc = 1
    if rc:
        print(
            "PREFLIGHT: FAIL — a covered command did not execute. The rest of this run "
            "would measure the reasoning loop instead of the product. Not starting.",
            flush=True,
        )
    return rc


async def _stream_probe(client, hadm_id: int) -> int:
    """One streaming turn, to prove the SSE path exists rather than assume it.

    The Reconciliation badge is the `prose_settled` stage, which only a caller that
    asked to watch prose arrive ever receives. The census is blocking, so this is the
    only place the badge path is exercised at all.

    Whether the badge *fired* is reported and deliberately not asserted: a clean
    answer legitimately has nothing to reconcile, so requiring the frame would fail
    every good turn. What is asserted is that the stream produced an answer — a
    stream that yields no frames is a broken transport, not a quiet turn.
    """
    print("PREFLIGHT: one streaming turn, checking the prose_settled stage …",
          flush=True)
    try:
        status, events = await client.stream(
            {"question": "/ehr", "hadm_id": hadm_id})
    except Exception as exc:  # transport-level
        print(f"PREFLIGHT: FAIL — the stream raised {type(exc).__name__}: {exc}",
              flush=True)
        return 1
    stages = [stage for stage, _ in events]
    settled = [data for stage, data in events if stage == "prose_settled"]
    print(f"  status={status} frames={len(events)} stages={sorted(set(stages))}",
          flush=True)
    print(f"  prose_settled: {'fired ' + str(settled) if settled else 'absent (clean turn)'}",
          flush=True)
    if status != 200 or not events:
        print("PREFLIGHT: FAIL — the streaming path produced no frames. The badge "
              "cannot be observed at all, so the evaluation would be silent about "
              "it. Not starting.", flush=True)
        return 1
    return 0


async def _preflight(client, args) -> int:
    """Smoke the deployed path before the long run.

    Two gates, because they fail differently. The smoke questions establish that
    retrieval answers from the corpus the deployment serves; when it does not,
    the serving configuration is wrong. The conversation establishes that the
    service accepts a replayed turn; when it does not, the harness is wrong, and
    every second turn of the run would be refused at the same step.
    """
    payload = json.loads(CASES.read_text())
    patients = payload["patients"][: args.preflight_n]
    # Cheapest gate first, and the one that actually failed last time. Retrieval
    # health cannot distinguish "the command answered" from "the reasoning loop
    # answered on the command's behalf", so the commands are asserted directly
    # before anything is measured.
    if await _route_coverage(client, patients[0]["hadm_id"]):
        return 1
    if await _stream_probe(client, patients[0]["hadm_id"]):
        return 1
    cases = [{"hadm_id": p["hadm_id"], "prompt": "meds",
              "question": question_for("meds", p["hadm_id"])} for p in patients]
    print(f"PREFLIGHT: {len(cases)} smoke questions against the deployment …",
          flush=True)

    recs = []
    for case in cases:
        rec = await _answer(client, case, 1, case["question"],
                            _body(case["question"], case))
        flags = classify_trace(rec)
        recs.append(rec)
        detail = rec.get("error") or ""
        print(f"  hadm={case['hadm_id']} status={rec.get('status')} "
              f"tools={[c.get('name') for c in rec.get('tool_calls') or []]} "
              f"missing_text={flags['missing_text']} "
              f"zero_passage={flags['zero_passage']}"
              + (f"  error={detail}" if detail else ""), flush=True)

    s = summarize(recs)
    r = _reachability(recs)
    print("  " + format_summary(s, "preflight"), flush=True)
    print(f"  preflight reachability: transport_error={r['transport_error']} "
          f"non_200={r['non_200']}", flush=True)
    # Both gates, in this order. Reachability first because it is the one that
    # makes the retrieval number meaningless: 0% retrieval failure across six
    # requests that never arrived is not a healthy run, it is no run.
    if r["unusable"]:
        print("PREFLIGHT: FAIL — the deployment did not answer. Nothing about "
              "retrieval can be concluded from this. Not starting the run.",
              flush=True)
        return 1
    if s["retrieval_fail_rate"] > args.max_fail_rate:
        print("PREFLIGHT: FAIL — retrieval is unhealthy against the deployment. "
              "Not starting the run.", flush=True)
        return 1

    # One whole conversation, with the risk chip, so the turn that gets replayed
    # is the turn that answers the follow-up. A second turn the service refuses is
    # a harness fault, and finding it here costs two requests instead of five
    # hundred and fifty.
    conv = {
        "hadm_id": patients[0]["hadm_id"],
        "prompt": "risk",
        "question": question_for("risk", patients[0]["hadm_id"]),
        "follow_up": follow_up_for("risk"),
    }
    print("PREFLIGHT: one whole conversation, to check the service accepts a "
          "replayed turn …", flush=True)
    first = await _answer(client, conv, 1, conv["question"],
                          _body(conv["question"], conv))
    if first.get("transport_error") or first.get("status") != 200:
        print(f"PREFLIGHT: FAIL — the first turn did not answer "
              f"(status {first.get('status')}: "
              f"{first.get('error') or 'no answer'}), so there is nothing to "
              f"replay and the second turn cannot be tested. Not starting the "
              f"run.", flush=True)
        return 1

    replayed = replay_turn(conv["question"], first)
    # Same builder as `_run_set`: one implementation, so a field cannot survive on one
    # path and go missing on the other.
    conv2 = _turn2_body(conv, replayed)
    second = await _answer(client, conv, 2, conv["follow_up"], conv2, prior=replayed)
    print(f"  turn 1: status={first.get('status')} "
          f"tools={[c.get('name') for c in first.get('tool_calls') or []]} | "
          f"replay carries {len(replayed['tool_calls'])} tool calls | "
          f"turn 2: status={second.get('status')} "
          f"tools={[c.get('name') for c in second.get('tool_calls') or []]}",
          flush=True)
    if second.get("transport_error") or second.get("status") != 200:
        print(f"PREFLIGHT: FAIL — the service did not accept the second turn "
              f"(status {second.get('status')}: "
              f"{second.get('error') or 'no answer'}). The replayed shape this "
              f"harness builds is not the shape the service accepts, so every "
              f"case would fail at its second turn. Not starting the run.",
              flush=True)
        return 1

    print("PREFLIGHT: PASS — retrieval is healthy and a replayed turn is "
          "accepted.", flush=True)
    return 0


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent-url", default=os.environ.get("AGENT_URL", ""),
                    help="deployed agent base URL (default: $AGENT_URL)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--adversarial-out", default=str(ADV_OUT))
    ap.add_argument("--prompt", choices=[*PROMPT_NAMES, "all"], default="all")
    ap.add_argument("--max-patients", type=int, default=None)
    ap.add_argument("--max-adversarial", type=int, default=None)
    ap.add_argument("--no-adversarial", action="store_true")
    ap.add_argument("--suite2", action="store_true",
                    help="also run the targeted modality matrix (suite2_cases.json)")
    ap.add_argument("--suite2-out", default=str(RESULTS / "suite2.jsonl"))
    ap.add_argument("--adversarial-only", action="store_true",
                    help="run only the adversarial probes (the adversarial job)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--preflight-n", type=int, default=6)
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--max-fail-rate", type=float, default=0.30)
    ap.add_argument("--upload-gcs", action="store_true",
                    help="copy this run's artifacts to gs://ecc-eval-artifacts")
    ap.add_argument("--run-tag", default=None,
                    help="label for this run, e.g. eval-pilot-101726")
    ap.add_argument("--check-every", type=int, default=20)
    args = ap.parse_args()

    if not args.agent_url:
        raise SystemExit("pass --agent-url or set AGENT_URL")

    clinical, adversarial = _build_cases(args)
    print(f"deployment: {args.agent_url}", flush=True)
    print(f"planned: {len(clinical)} clinical + {len(adversarial)} adversarial "
          f"= {len(clinical) + len(adversarial)} cases "
          f"(concurrency {args.concurrency})", flush=True)

    client = AgentClient(args.agent_url, _identity_token(args.agent_url))
    try:
        if not args.skip_preflight:
            rc = await _preflight(client, args)
            if rc != 0:
                return rc
        if args.preflight_only:
            return 0

        rc = 0
        if clinical:
            rc |= await _run_set(client, clinical, Path(args.out), args, "clinical")
        if args.suite2 and rc == 0:
            suite2 = _suite2_cases(args)
            print(f"planned: {len(suite2)} suite-2 modality cases", flush=True)
            rc |= await _run_set(client, suite2, Path(args.suite2_out), args,
                                 "suite2", gate_retrieval=False)
        if adversarial and rc == 0:
            rc |= await _run_set(client, adversarial, Path(args.adversarial_out),
                                 args, "adversarial", gate_retrieval=False)

        # Final gate: the caller should not judge a run whose retrieval was
        # broken for a large share of it.
        for path, label, gate_retrieval in (
                (Path(args.out), "clinical", True),
                (Path(args.adversarial_out), "adversarial", False)):
            recs = []
            if path.exists():
                for line in path.read_text().splitlines():
                    if line.strip():
                        try:
                            recs.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            s = summarize(recs)
            r = _reachability(recs)
            print("  " + format_summary(s, f"final {label}"), flush=True)
            print(f"  final {label} reachability: "
                  f"transport_error={r['transport_error']} "
                  f"non_200={r['non_200']}", flush=True)
            if r["n"] and r["unusable_rate"] > args.max_fail_rate:
                print(f"FINAL GATE: {label} — {r['unusable']} of {r['n']} requests "
                      f"produced no answer. Do not judge this run.", flush=True)
                rc |= 1
            if gate_retrieval and s["n"] and s["retrieval_fail_rate"] > args.max_fail_rate:
                print(f"FINAL GATE: {label} retrieval-failure rate above "
                      f"threshold — do not judge this run.", flush=True)
                rc |= 1
        return rc
    finally:
        await client.aclose()
        _upload_artifacts(args)


def _upload_artifacts(args) -> None:
    """Copy this run's artifacts to the bucket when `--upload-gcs` was given.

    Called from `finally` so a run that fails the final gate still ships what it did
    collect: the partial run is exactly what a failure investigation needs.
    """
    if not getattr(args, "upload_gcs", False):
        return
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from storage import sync_eval_to_gcs  # noqa: PLC0415

        tag = getattr(args, "run_tag", None) or "unlabelled"
        sent = sync_eval_to_gcs(tag, [Path(args.out), Path(args.adversarial_out)])
        print(f"  artifacts: {len(sent)} uploaded to gs://ecc-eval-artifacts/runs/{tag}/"
              if sent else "  artifacts: upload skipped (see log)", flush=True)
    except Exception:  # noqa: BLE001 — a copy must never fail the run
        print("  artifacts: upload skipped (see log)", flush=True)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
