"""The session's evidence, as a record rather than a transcript.

Until now a turn's evidence reached the next one by being reconstructed from the stored
conversation: `graph.replay_messages` re-resolved each prior call's payload, `entry`-level
counter sums recomputed the numbering, and `postprocess/answer` re-derived which sections
had been read. Three consumers, three derivations, one stored prose — and the audit of
revision `c8bee13` found the consequences (`29-forensic_audit_6_failures_2026-11-05.md`):
a turn that answered correctly from a replayed lab panel was sent back to search the notes,
because nothing on the state said the panel was already there.

This module is the record that removes the need to infer. One `SessionEvidenceState` per
conversation, bound to one admission, append-only:

  * every citable unit a tool returned is registered once, under a permanent 1-based
    `ref_id`, and re-registering the same evidence returns the id it already has;
  * which note sections have been read, and which were asked for and came back empty, is
    recorded — so a later turn can say "this record has no discharge medication list"
    from a finding rather than from a fresh guess;
  * the readmission estimate and the published records are held beside the numbered units,
    because neither publishes a position (`^[PMID: …]` names a record; the estimate is the
    model's own output and the rubric scores it as unmarked).

**Law 1 (citation ownership).** What counts as a citable unit is not decided here. That is
`postprocess/citations.citable_passages`' answer, and this module asks it — one unit per
note passage, one per laboratory analyte, nothing for a prediction or a search result set.
The `ref_id` this module assigns is the position `citations` produced, in the order it
produced it, which is what makes the session's numbering and a turn's numbering the same
sequence by construction rather than by agreement.

**Law 4 (tool registries).** The tool names this module recognises are `routing`'s, read
from its own table. Nothing here re-lists them.

**The array.** `ordered_citable_units` is the session's positional array — position `n` is
`items[n]` by construction — and it is what `citations.EvidenceLedger` and
`presentation.resolve_sources` index, so a marker the model wrote and the record it opens
are the same position by construction rather than by agreement.

It imports nothing above `postprocess/` — only `routing` and `citations`, both of which
`postprocess/` may reach.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from services.agent.routing import routing
from services.agent.postprocess import citations

#: The laboratory section label `citations.lab_passage` gives an analyte. It is the
#: section a lab marker opens onto, and this module keeps the analyte's own key beside it.
_LAB_TOOL = "get_lab_trends"
_PREDICTION_TOOL = "predict_readmission"

#: The two sections a `discharge_medications` call settles between them. The tool returns
#: whichever of them the note has, and a missing one is a finding about the record.
_MEDICATION_SECTIONS: tuple[str, str] = ("discharge_medications", "discharge_instructions")

#: Keys carried onto `prediction` from the payload. A summary rather than a copy: the
#: estimate is this turn's answer, not evidence a reader can open, and what a later turn
#: needs from it is the figures and the attribution.
_PREDICTION_KEYS: tuple[str, ...] = (
    "probability", "threshold", "decision", "base_value", "top_factors",
    "model_version", "feature_source", "probability_display", "threshold_display",
)

PRESENT = "present"
ABSENT = "absent"

#: The `scope` a `search_literature` call declares when it is serving the comparison
#: table rather than the answer. Mirrors `citations._searched_for_evidence`, which owns
#: the rule; the literal is repeated here because `citations` publishes the predicate
#: rather than the scope name, and `postprocess` may not grow a second policy module.
_CANDIDATE_POOL_SCOPE = "readmission"

#: The tags `render_for_prompt` builds the record out of. A passage, a section name, a
#: factor label or a record title is written BETWEEN them, so a value carrying one of
#: them could close the region it sits in and make everything after it read as
#: instruction rather than as the patient's chart.
#:
#: The escape lives here, beside the renderer, for the same reason `citations` owns the
#: marker regex: the module that PRINTS a construct owns the escape for it. It used to
#: live in `engine/mcp_client.py`, applied to tool payloads on their way into the record,
#: which protected exactly the strings that came through that one caller — not the
#: section names `_settle_sections` takes from the model's own arguments, and not any
#: future caller of `render_for_prompt` at all (audit 2026-11-05, §2.3).
_RECORD_TAG_RE = re.compile(
    r"<\s*/?\s*(?:session_evidence_state|evidence_items|section_inventory"
    r"|readmission_risk|literature)\b",
    re.IGNORECASE,
)


def _escape(value: Any) -> str:
    """One value as it may be written inside the record.

    `<` becomes `&lt;`, so the tag is still READABLE — a note that genuinely contains the
    string `</evidence_items>` is quoted rather than silently altered — while no longer
    being a tag. The same neutralisation `_escape_delimiter` applies to the tool-result
    wrapper, applied to the tags this renderer prints.
    """
    return _RECORD_TAG_RE.sub(lambda match: "&lt;" + match.group(0)[1:], str(value))


def _is_candidate_pool(tool_args: Mapping[str, Any] | None) -> bool:
    """Whether this literature search served the comparison table rather than the answer.

    `search_literature` has two callers with different questions. The comparison table
    asks for the candidate cohort a condition might be measured against; the model asks
    what the literature says. Both arrive under one tool name, and only the second is
    evidence a turn may answer from — `citations._searched_for_evidence` is the citation
    owner's statement of the same rule.

    A call that declares no scope searched for evidence: `general` is the tool's own
    default, and every caller before the comparator work left it unset.
    """
    scope = str((tool_args or {}).get("scope") or "").strip().lower()
    return scope == _CANDIDATE_POOL_SCOPE


def _digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def recognized_tool(name: str) -> bool:
    """Whether `routing` advertises this tool at all.

    `TOOL_LABELS` is the one table with a row per advertised tool, and a test walks the
    live MCP tool list against it, so this is Law 4's registry read rather than a second
    copy of it.
    """
    return str(name or "") in routing.TOOL_LABELS


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    """One registered piece of evidence.

    `ref_id` is the session's own position, assigned once and never reused. It is `None`
    for evidence that publishes no position — a published record, cited by PMID — and for
    anything registered before numbering exists.
    """

    ref_id: int | None
    hadm_id: int | None
    source_tool: str
    #: The note section, or the analyte key for a laboratory unit.
    section: str | None
    #: What the unit says, verbatim: the passage text, the trajectory as `citations`
    #: formats it, or the abstract of a published record.
    text: str
    #: Carried from the payload, never inferred. Only `get_lab_trends` sets it today.
    synthetic: bool
    pmid: str | None
    #: Provenance and nothing else: where the unit came from and what it was read
    #: against. JSON-safe, because the state is serialized.
    metadata: Mapping[str, Any]
    #: Deterministic identity, so the same evidence retrieved twice is one item.
    dedup_key: str
    #: The unit exactly as `citations.citable_passages` emitted it, for an item that took
    #: a position. Held verbatim rather than rebuilt from the fields above: this module
    #: does not decide what a unit is (Law 1), and it does not restate it either. Rebuilding
    #: it here would put `id`, `score`, `granularity`, `cite_source` and `cite_title` — the
    #: fields a reader's row is drawn from — in a second implementation that could drift
    #: from the first.
    citable: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class IngestResult:
    """What one tool call did to the state."""

    #: Positions assigned by this call.
    assigned: tuple[int, ...] = ()
    #: Positions this call re-stated, already present before it ran.
    reused: tuple[int, ...] = ()
    #: The position of every citable unit this call carried, in payload order —
    #: assigned or reused, one entry per unit. `assigned` and `reused` say what happened
    #: to the numbering; this says what the numbering IS. A renderer stamps the copy the
    #: model reads from here rather than counting the payload again from an offset, which
    #: is the difference between a re-retrieved passage keeping its position and getting
    #: a second one.
    slot_ids: tuple[int, ...] = ()
    #: Whether the call was turned away: an error payload, a tool this deployment does
    #: not advertise, or an admission that is not the session's.
    refused: bool = False
    #: Section keys this call settled, as `section` or `section: present|absent`.
    checked: tuple[str, ...] = ()
    #: Analytes this call settled, as `analyte: present|absent`. Separate from `checked`
    #: because the two inventories answer different questions and a reader of one must not
    #: have to filter the other: a note section and a laboratory measurement are not the
    #: same kind of thing to look for.
    lab_checked: tuple[str, ...] = ()


@dataclass
class SessionEvidenceState:
    """One conversation's evidence, bound to one admission, append-only.

    The ordering rules are the whole contract:

      * `items` is keyed by a 1-based position that only ever grows. A deduplicated
        re-ingest returns the position it already had, so a marker written in an earlier
        turn still addresses the passage it was written for.
      * `checked_sections` records what has been READ, including sections that came back
        empty. Presence is monotonic: once a section has been returned, a later call that
        omits it cannot downgrade it to absent.
    """

    hadm_id: int
    items: dict[int, EvidenceItem] = field(default_factory=dict)
    checked_sections: dict[str, str] = field(default_factory=dict)
    #: Analytes this session has read values for, and analytes it asked for and did not
    #: get. The laboratory twin of `checked_sections`, and the reason it exists: the model
    #: was asked for hemoglobin and creatinine, read one of them, and then stated that the
    #: record did not document the other (2026-11-06) — an absence about a value nobody
    #: had looked for. An inventory cannot be talked around the way prose can.
    checked_labs: dict[str, str] = field(default_factory=dict)
    literature: dict[str, EvidenceItem] = field(default_factory=dict)
    prediction: dict[str, Any] | None = None
    #: `dedup_key` -> `ref_id`, derived state. Rebuilt on load rather than trusted.
    _dedup_index: dict[str, int] = field(default_factory=dict, repr=False)
    _next_ref: int = 1

    # -- ingestion ---------------------------------------------------------------

    def ingest_tool_output(
        self,
        tool_name: str,
        tool_args: Mapping[str, Any] | None,
        raw_output: Any,
    ) -> IngestResult:
        """Register one tool call's result, or refuse it.

        Refusal is total: a refused call leaves the state exactly as it was and does not
        advance the counter. That is the difference between a call that produced evidence
        and a call that was turned away, and a counter that moved on a refusal would leave
        a gap in the numbering that no marker could address.
        """
        if not recognized_tool(tool_name):
            return IngestResult(refused=True)
        if not isinstance(raw_output, Mapping):
            return IngestResult(refused=True)
        if raw_output.get("error"):
            # `patient_scope_refused`, `tool_already_spent`, a tool's own failure: the
            # call happened and produced nothing, which is not evidence.
            return IngestResult(refused=True)
        if not self._same_admission(tool_args, raw_output):
            return IngestResult(refused=True)

        name = str(tool_name)
        if name in routing.LITERATURE_TOOLS:
            if _is_candidate_pool(tool_args):
                # The `/ehr` comparison table searches PubMed for the candidate cohort a
                # condition might be compared against, and that pool is not this answer's
                # evidence — `citations._searched_for_evidence` excludes it for exactly
                # this reason. Filing it here would make the record claim the thread had
                # read published evidence it never rested on, and a later literature
                # question would be answered from a candidate list instead of searched
                # for (audit 2026-11-05, §2.3 carry-forward).
                return IngestResult()
            return self._ingest_literature(name, raw_output)
        if name == _PREDICTION_TOOL:
            return self._ingest_prediction(name, raw_output)
        if name == _LAB_TOOL:
            return self._ingest_lab(name, raw_output, tool_args)
        if name in routing.PASSAGE_TOOLS:
            return self._ingest_passages(name, raw_output, tool_args)
        # A recognised tool whose result this record does not index — the pre-flight
        # profile. Refusing it would be wrong (it ran) and registering it would be
        # wrong (it publishes no unit); it is simply not evidence.
        return IngestResult()

    def _same_admission(self, tool_args: Mapping[str, Any] | None,
                        raw_output: Mapping[str, Any]) -> bool:
        """Whether this payload belongs to the session's admission.

        Defence in depth behind the outbound guard: that guard refuses the CALL, this
        refuses the RECORD. A state that could be fed another admission's passages would
        be a state whose numbering a reader cannot trust, and it would be persisted.
        """
        for source in (tool_args, raw_output):
            if not isinstance(source, Mapping):
                continue
            hadm = source.get("hadm_id")
            if hadm is None:
                continue
            try:
                if int(hadm) != int(self.hadm_id):
                    return False
            except (TypeError, ValueError):
                return False
        return True

    def _register(self, *, source_tool: str, section: str | None, text: str,
                  synthetic: bool, pmid: str | None,
                  metadata: Mapping[str, Any], dedup_key: str,
                  citable: Mapping[str, Any] | None = None,
                  positional: bool = True) -> tuple[int | None, bool]:
        """Assign or reuse the position for one unit. Returns `(ref_id, reused)`."""
        if not positional:
            item = EvidenceItem(
                ref_id=None, hadm_id=self.hadm_id, source_tool=source_tool,
                section=section, text=text, synthetic=synthetic, pmid=pmid,
                metadata=dict(metadata), dedup_key=dedup_key,
            )
            key = pmid or dedup_key
            self.literature[key] = item
            return None, False
        existing = self._dedup_index.get(dedup_key)
        if existing is not None:
            # The FIRST registration keeps the unit. A re-retrieval may present the same
            # passage with a different score or a different query, and rewriting the item
            # would make a reader's row change under a marker that already pointed at it.
            return existing, True
        ref_id = self._next_ref
        self._next_ref += 1
        self.items[ref_id] = EvidenceItem(
            ref_id=ref_id, hadm_id=self.hadm_id, source_tool=source_tool,
            section=section, text=text, synthetic=synthetic, pmid=pmid,
            metadata=dict(metadata), dedup_key=dedup_key,
            citable=dict(citable or {}),
        )
        self._dedup_index[dedup_key] = ref_id
        return ref_id, False

    def _ingest_passages(self, name: str, raw_output: Mapping[str, Any],
                         tool_args: Mapping[str, Any] | None) -> IngestResult:
        units = citations.citable_passages(raw_output)
        assigned: list[int] = []
        reused: list[int] = []
        slot_ids: list[int] = []
        sections_seen: list[str] = []
        for unit in units:
            section = str(unit.get("section") or "").strip() or None
            text = str(unit.get("text") or "")
            passage_id = str(unit.get("id") or "").strip()
            key = (
                f"note:{passage_id}" if passage_id
                else f"note:{self.hadm_id}:{section}:{_digest(text)}"
            )
            metadata = {"section": section or ""}
            if passage_id:
                metadata["passage_id"] = passage_id
            for extra in ("retrieval", "granularity", "query", "medication_source"):
                value = unit.get(extra) if extra in unit else raw_output.get(extra)
                if isinstance(value, str) and value.strip():
                    metadata[extra] = value.strip()
            ref_id, was_reused = self._register(
                source_tool=name, section=section, text=text, synthetic=False,
                pmid=None, metadata=metadata, dedup_key=key, citable=unit,
            )
            if section:
                sections_seen.append(section)
            if ref_id is not None:
                (reused if was_reused else assigned).append(ref_id)
                slot_ids.append(ref_id)

        checked = self._settle_sections(name, sections_seen, tool_args)
        return IngestResult(assigned=tuple(assigned), reused=tuple(reused),
                            slot_ids=tuple(slot_ids), refused=False, checked=checked)

    def _ingest_lab(self, name: str, raw_output: Mapping[str, Any],
                    tool_args: Mapping[str, Any] | None) -> IngestResult:
        units = citations.citable_passages(raw_output)
        synthetic = bool(raw_output.get("synthetic", False))
        checked_labs: list[str] = []
        trends = [trend for trend in (raw_output.get("trends") or [])
                  if isinstance(trend, Mapping)]
        # `citable_passages` maps one analyte to one unit in the order the panel gave
        # them, so the analyte's own key is read from the same position. When the two
        # lists disagree the key falls back to the unit's text rather than to an index
        # that might pair the wrong analyte with the wrong series.
        assigned: list[int] = []
        reused: list[int] = []
        slot_ids: list[int] = []
        for position, unit in enumerate(units):
            trend = trends[position] if position < len(trends) else {}
            analyte = str(trend.get("key") or "").strip() or None
            text = str(unit.get("text") or "")
            if analyte:
                key = f"lab:{self.hadm_id}:{analyte}"
            else:
                key = f"lab:{self.hadm_id}:{_digest(text)}"
            metadata: dict[str, Any] = {}
            for field_name in ("unit", "reference_low", "reference_high", "decimals"):
                if field_name in trend:
                    metadata[field_name] = trend[field_name]
            flags = sorted({
                str(point.get("flag") or "").strip()
                for point in (trend.get("series") or [])
                if isinstance(point, Mapping) and str(point.get("flag") or "").strip()
            })
            if flags:
                metadata["flags"] = ",".join(flags)
            ref_id, was_reused = self._register(
                source_tool=name, section=analyte, text=text, synthetic=synthetic,
                pmid=None, metadata=metadata, dedup_key=key, citable=unit,
            )
            if ref_id is not None:
                (reused if was_reused else assigned).append(ref_id)
                slot_ids.append(ref_id)
            if analyte:
                checked_labs.extend(self._mark_lab(analyte, PRESENT))
        # An analyte the call ASKED for and did not get back is the one absence this tool
        # can establish: the measurement was requested of this admission's record and the
        # record returned nothing for it. The request is read from the call's own
        # arguments, never inferred from the panel: a panel that omits a value the caller
        # never named says nothing about that value.
        requested = ""
        if isinstance(tool_args, Mapping):
            requested = str(tool_args.get("analyte") or "").strip()
        if requested and requested.lower() not in {
                key.lower() for key in self.checked_labs}:
            checked_labs.extend(self._mark_lab(requested, ABSENT))
        return IngestResult(assigned=tuple(assigned), reused=tuple(reused),
                            slot_ids=tuple(slot_ids), refused=False, checked=(),
                            lab_checked=tuple(checked_labs))

    def _ingest_literature(self, name: str,
                           raw_output: Mapping[str, Any]) -> IngestResult:
        """Published records. They take no position — a badge names a record, not a slot.

        `citations.citable_passages` already returns nothing for an article payload, and
        that is the same rule stated once more here: numbering a PMID beside the note
        passages would give one glyph two meanings.
        """
        for article in raw_output.get("articles") or []:
            if not isinstance(article, Mapping):
                continue
            pmid = str(article.get("pmid") or "").strip()
            if not pmid:
                continue
            metadata = {
                key: str(article.get(key) or "").strip()
                for key in ("title", "journal", "pub_date", "url")
                if article.get(key)
            }
            self._register(
                source_tool=name, section=None,
                text=str(article.get("abstract") or ""), synthetic=False, pmid=pmid,
                metadata=metadata, dedup_key=f"lit:{pmid}", positional=False,
            )
        return IngestResult()

    def _ingest_prediction(self, name: str,
                           raw_output: Mapping[str, Any]) -> IngestResult:
        """The estimate, kept beside the units because it publishes none of them."""
        self.prediction = {
            key: raw_output[key] for key in _PREDICTION_KEYS if key in raw_output
        }
        self.checked_sections[name] = PRESENT
        return IngestResult(checked=(name,))

    # -- section inventory -------------------------------------------------------

    def _mark(self, section: str, state: str) -> list[str]:
        """Record a section's state, never downgrading `present` to `absent`."""
        name = str(section or "").strip()
        if not name:
            return []
        current = self.checked_sections.get(name)
        if current == PRESENT:
            return []
        if current == state:
            return []
        self.checked_sections[name] = state
        return [f"{name}: {state}"]

    def _mark_lab(self, analyte: str, state: str) -> list[str]:
        """Record an analyte's state. Same monotonicity as `_mark`, for the same reason.

        Keys are compared case-insensitively: a note writes `HbA1c` and a panel writes
        `hba1c`, and an inventory that treated those as two measurements would let one
        reading license an absence about the other.
        """
        name = str(analyte or "").strip().lower()
        if not name:
            return []
        current = self.checked_labs.get(name)
        if current == PRESENT or current == state:
            return []
        self.checked_labs[name] = state
        return [f"{name}: {state}"]

    def _settle_sections(self, name: str, seen: Iterable[str],
                         tool_args: Mapping[str, Any] | None) -> tuple[str, ...]:
        returned = [section for section in seen if section]
        changed: list[str] = []
        for section in returned:
            changed.extend(self._mark(section, PRESENT))

        if name == "discharge_medications":
            # The call asks for two sections and returns whichever the note has, so an
            # omission here is a statement about the record: this note has no discharge
            # medication list, or no discharge instructions.
            for section in _MEDICATION_SECTIONS:
                if section not in returned:
                    changed.extend(self._mark(section, ABSENT))
            return tuple(changed)

        requested: list[str] = []
        if isinstance(tool_args, Mapping):
            raw = tool_args.get("sections")
            if isinstance(raw, (list, tuple)):
                requested.extend(str(entry) for entry in raw)
            single = tool_args.get("section")
            if isinstance(single, str):
                requested.extend(single.split(","))
        for section in requested:
            cleaned = str(section or "").strip()
            if cleaned and cleaned not in returned:
                changed.extend(self._mark(cleaned, ABSENT))
        return tuple(changed)

    # -- hydration ---------------------------------------------------------------

    @classmethod
    def hydrate_from_calls(
        cls,
        hadm_id: int,
        *call_groups: Iterable[Mapping[str, Any]],
    ) -> "SessionEvidenceState":
        """Build a state from call records, in the order the model was shown them.

        Two shapes reach this: `{name, args, response}` from `RecordedToolCall`, and
        `{name, args, payload}` from a replayed turn's stored transcript. Both are read,
        because both describe the same fact and neither caller should have to translate.
        """
        state = cls(hadm_id=hadm_id)
        for group in call_groups:
            for call in group or []:
                if not isinstance(call, Mapping):
                    continue
                output = call.get("response")
                if output is None:
                    output = call.get("payload")
                args = call.get("args")
                state.ingest_tool_output(
                    str(call.get("name") or ""),
                    args if isinstance(args, Mapping) else None,
                    output,
                )
        return state

    # -- projection --------------------------------------------------------------

    def ordered_citable_units(self) -> list[dict[str, Any]]:
        """The positional units, in `1..K` `ref_id` order.

        This is the array a marker indexes: `citations.EvidenceLedger` builds its units
        from it and `presentation.resolve_sources` resolves position `n` at `[n - 1]`.
        Position `n` is `items[n]` by construction — the numbering only ever grows and a
        refused call never takes a position — and that is asserted rather than assumed,
        because a hole here would shift every unit after it and resolution would still
        appear to work.

        The difference from concatenating the tool results is the whole point of the
        sprint: a passage retrieved in turn 1 and again in turn 2 occupies ONE position
        here and two in a concatenation. The turn that re-retrieves it writes a marker
        that addresses the position the record already gave it, so the marker written in
        the first turn and the marker written in the second name the same evidence.
        """
        units: list[dict[str, Any]] = []
        for ref_id in range(1, self._next_ref):
            item = self.items.get(ref_id)
            if item is None:
                raise ValueError(
                    f"evidence position {ref_id} is missing from 1..{self._next_ref - 1}")
            units.append(dict(item.citable))
        return units

    def position_count(self) -> int:
        """`K` in `1..K`: how many positions this record has assigned.

        The one count of them. `citations.citation_bound` reaches it through
        `ordered_citable_units()`; anything else that needs the length asks here rather
        than adding up payloads.
        """
        return len(self.items)

    def position_of_note_passage(self, passage_id: str) -> int | None:
        """The position a note passage holds, by the tool's own id, or None.

        The lookup a caller uses when it must name the position of a passage it knows by
        id rather than by order — `/meds` writes one sentence about one passage, and after
        an `/ehr` that passage may already be at position 4. The key is the same one
        `_ingest_passages` deduplicates on, so this cannot disagree with the numbering:
        it reads the index that numbering is built from.

        `None` means the passage is not in the record — a turn with no admission, or a
        call that was refused. A caller that gets None writes no marker rather than
        guessing one.
        """
        cleaned = str(passage_id or "").strip()
        if not cleaned:
            return None
        return self._dedup_index.get(f"note:{cleaned}")

    # -- rendering ---------------------------------------------------------------

    def render_for_prompt(self) -> str:
        """The state as the model reads it. Deterministic for a given state.

        Every interpolated value passes through `_escape`: this is the one place the
        record becomes text the model reads, so it is the one place the escape has to
        hold. Nothing is escaped on the way IN — `to_dict()` keeps the raw strings and a
        `dedup_key` hashes what the tool actually said, because escaping is a property of
        the rendering and not of the evidence.

        The header carries no admission id. The record is bound to one at construction and
        every unit in it is already scoped, so the number is not evidence the model needs —
        and it is the one value it must never write back. Measured 2026-11-06: with the id
        in the window, a free-text turn called
        `discharge_medications({"hadm_id": "90000069"})` and the injector had to decide what
        to do with an argument the model had no business supplying.
        """
        lines = ["<session_evidence_state>"]
        lines.append("<evidence_items>")
        for ref_id in sorted(self.items):
            item = self.items[ref_id]
            lines.append(self._item_header(item))
            if item.text.strip():
                lines.append(_escape(item.text.strip()))
        lines.append("</evidence_items>")

        lines.append("<section_inventory>")
        for section in sorted(self.checked_sections):
            lines.append(
                f"{_escape(section)}: {_escape(self.checked_sections[section])}")
        lines.append("</section_inventory>")

        # The laboratory inventory, when there is one. Separate from the note sections
        # because it answers a different question — which VALUES this session asked about
        # and what came back — and because an analyte that was never read is not the same
        # statement as a note section that was read and held nothing.
        if self.checked_labs:
            lines.append("<lab_inventory>")
            for analyte in sorted(self.checked_labs):
                lines.append(f"{_escape(analyte)}: {_escape(self.checked_labs[analyte])}")
            lines.append("</lab_inventory>")

        risk = self._render_prediction()
        if risk:
            lines.append(risk)
        literature = self._render_literature()
        if literature:
            lines.append(literature)
        lines.append("</session_evidence_state>")
        return "\n".join(lines)

    @staticmethod
    def _item_header(item: EvidenceItem) -> str:
        """`[^[k] | tool=… | section=…]` — the provenance a marker resolves through.

        `section` is the analyte key or the note heading, and the heading reaches the
        record from the tool's own passage. It is escaped like every other value written
        between the record's tags.

        `medication_source` rides along on a `discharge_medications` passage because the
        distinction it states is the one a follow-up is most likely to get wrong: a drug
        in a `dedicated` passage is on the record's discharge list, one in a `narrative`
        passage is a drug the course or the plan mentions and may have been given on the
        ward. Stated here rather than left in `metadata`, which the model never reads.
        """
        parts = [f"[^[{item.ref_id}]", f"tool={_escape(item.source_tool)}"]
        if item.section:
            parts.append(f"section={_escape(item.section)}")
        provenance = item.metadata.get("medication_source")
        if isinstance(provenance, str) and provenance.strip():
            parts.append(f"medication_source={_escape(provenance.strip())}")
        if item.synthetic:
            parts.append("synthetic=true")
        return " | ".join(parts) + "]"

    def _render_prediction(self) -> str:
        if not self.prediction:
            return ""
        lines = ["<readmission_risk>"]
        for key in ("probability", "probability_display", "threshold",
                    "threshold_display", "decision"):
            if key in self.prediction:
                lines.append(f"{key}: {_escape(self.prediction[key])}")
        factors = self.prediction.get("top_factors")
        if isinstance(factors, list) and factors:
            lines.append("top_factors:")
            for factor in factors:
                if not isinstance(factor, Mapping):
                    continue
                parts = [f"feature={_escape(factor.get('feature'))}"]
                for key in ("contribution", "direction", "value"):
                    if key in factor:
                        parts.append(f"{key}={_escape(factor[key])}")
                lines.append("  " + " | ".join(parts))
        lines.append("</readmission_risk>")
        return "\n".join(lines)

    def _render_literature(self) -> str:
        if not self.literature:
            return ""
        lines = ["<literature>"]
        for pmid in sorted(self.literature):
            item = self.literature[pmid]
            header = [f"[PMID: {_escape(pmid)}"]
            for key in ("title", "journal", "pub_date"):
                value = item.metadata.get(key)
                if value:
                    header.append(f"{key}={_escape(value)}")
            lines.append(" | ".join(header) + "]")
            if item.text.strip():
                lines.append(_escape(item.text.strip()))
        lines.append("</literature>")
        return "\n".join(lines)

    # -- serialization -----------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """A pure JSON-able view of the state, positions included."""
        return {
            "hadm_id": self.hadm_id,
            "next_ref": self._next_ref,
            "items": {str(ref_id): _item_to_dict(item)
                      for ref_id, item in sorted(self.items.items())},
            "checked_sections": dict(sorted(self.checked_sections.items())),
            "checked_labs": dict(sorted(self.checked_labs.items())),
            "literature": {pmid: _item_to_dict(item)
                           for pmid, item in sorted(self.literature.items())},
            "prediction": self.prediction,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SessionEvidenceState":
        """Rebuild a state, refusing anything that is not one.

        Strict on purpose: this is the object a session is resumed from, and a state that
        loads half-corrupted is worse than one that fails — the numbering would silently
        address the wrong evidence, which is the failure the whole module exists to
        prevent.
        """
        if not isinstance(data, Mapping):
            raise ValueError("evidence state must be an object")
        hadm_id = data.get("hadm_id")
        if not isinstance(hadm_id, int) or isinstance(hadm_id, bool):
            raise ValueError("evidence state has no valid hadm_id")
        next_ref = data.get("next_ref")
        if not isinstance(next_ref, int) or isinstance(next_ref, bool) or next_ref < 1:
            raise ValueError("evidence state has no valid next_ref")

        state = cls(hadm_id=hadm_id, _next_ref=next_ref)

        raw_items = data.get("items") or {}
        if not isinstance(raw_items, Mapping):
            raise ValueError("evidence state items must be an object")
        for key, raw in raw_items.items():
            try:
                ref_id = int(key)
            except (TypeError, ValueError):
                raise ValueError(f"evidence item key {key!r} is not a number") from None
            item = _item_from_dict(raw)
            if item.ref_id != ref_id:
                raise ValueError(
                    f"evidence item key {key!r} does not match its ref_id {item.ref_id}")
            if ref_id < 1 or ref_id >= next_ref:
                raise ValueError(
                    f"evidence item {ref_id} is outside 1..{next_ref - 1}")
            if item.dedup_key in state._dedup_index:
                raise ValueError(f"duplicate dedup_key {item.dedup_key!r}")
            state.items[ref_id] = item
            state._dedup_index[item.dedup_key] = ref_id

        raw_sections = data.get("checked_sections") or {}
        if not isinstance(raw_sections, Mapping):
            raise ValueError("checked_sections must be an object")
        for section, value in raw_sections.items():
            if value not in (PRESENT, ABSENT):
                raise ValueError(
                    f"checked_sections[{section!r}] must be 'present' or 'absent'")
            state.checked_sections[str(section)] = value

        # Optional, because a state written before the laboratory inventory existed is
        # still a valid state: it holds no lab facts and says so by having none.
        raw_labs = data.get("checked_labs") or {}
        if not isinstance(raw_labs, Mapping):
            raise ValueError("checked_labs must be an object")
        for analyte, value in raw_labs.items():
            if value not in (PRESENT, ABSENT):
                raise ValueError(
                    f"checked_labs[{analyte!r}] must be 'present' or 'absent'")
            state.checked_labs[str(analyte)] = value

        raw_literature = data.get("literature") or {}
        if not isinstance(raw_literature, Mapping):
            raise ValueError("literature must be an object")
        for pmid, raw in raw_literature.items():
            item = _item_from_dict(raw)
            if item.ref_id is not None:
                raise ValueError(f"published record {pmid!r} must not carry a ref_id")
            state.literature[str(pmid)] = item

        prediction = data.get("prediction")
        if prediction is not None and not isinstance(prediction, Mapping):
            raise ValueError("prediction must be an object or null")
        state.prediction = dict(prediction) if prediction is not None else None
        return state


def _item_to_dict(item: EvidenceItem) -> dict[str, Any]:
    return {
        "ref_id": item.ref_id,
        "hadm_id": item.hadm_id,
        "source_tool": item.source_tool,
        "section": item.section,
        "text": item.text,
        "synthetic": item.synthetic,
        "pmid": item.pmid,
        "metadata": dict(item.metadata),
        "dedup_key": item.dedup_key,
        "citable": dict(item.citable),
    }


def _item_from_dict(raw: Any) -> EvidenceItem:
    if not isinstance(raw, Mapping):
        raise ValueError("evidence item must be an object")
    ref_id = raw.get("ref_id")
    if ref_id is not None and (not isinstance(ref_id, int) or isinstance(ref_id, bool)):
        raise ValueError("evidence item ref_id must be a whole number or null")
    metadata = raw.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        raise ValueError("evidence item metadata must be an object")
    dedup_key = raw.get("dedup_key")
    if not isinstance(dedup_key, str) or not dedup_key:
        raise ValueError("evidence item has no dedup_key")
    source_tool = raw.get("source_tool")
    if not isinstance(source_tool, str) or not source_tool:
        raise ValueError("evidence item has no source_tool")
    citable = raw.get("citable") or {}
    if not isinstance(citable, Mapping):
        raise ValueError("evidence item citable unit must be an object")
    return EvidenceItem(
        ref_id=ref_id,
        hadm_id=raw.get("hadm_id"),
        source_tool=source_tool,
        section=raw.get("section"),
        text=str(raw.get("text") or ""),
        synthetic=bool(raw.get("synthetic", False)),
        pmid=raw.get("pmid"),
        metadata=dict(metadata),
        dedup_key=dedup_key,
        citable=dict(citable),
    )
