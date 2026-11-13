"""Sprint 02: the selection slice's contract, proven offline.

Fourteen scenarios over five unused records, two per trajectory. These tests are the
acceptance for the slice: its cohort is disjoint from every reserved record, its shape is
what `task.py` and `judge_call.py` consume, its ground truth is declared and non-empty, and
nothing in the path that builds it can call a model.

The last group proves the sync script end to end against a mock Langfuse client, because the
real one is not reachable from this shell — and a sync that has never run is a script nobody
knows works.
"""

import ast
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from evaluation.platform import judge_call  # noqa: E402
from evaluation.platform import selection_slice as suite  # noqa: E402
from evaluation.platform import sync_selection_slice  # noqa: E402
from evaluation.platform.schemas import DatasetItemInput, MAX_TURNS  # noqa: E402
from evaluation.platform.task import placeholder_turns, resolve_prompt  # noqa: E402
from evaluation.outpatient_oracle import OUTPATIENT_HADM_IDS  # noqa: E402

ROWS = suite.read_rows()
BY_ID = {row["id"]: row for row in ROWS}

STUB_MANIFEST = {
    "encounter": "inpatient",
    "discharge_location": "Home",
    "admission_type": "Emergency (EW EMER.)",
    "age": "65",
    "gender": "Male",
    "index_los_days": "2",
    "prior_inpatient_days": "99.2083",
}


def _manifest_for(hadm_ids):
    return {int(hadm): dict(STUB_MANIFEST) for hadm in hadm_ids}


# --------------------------------------------------------------------------------------
# 1. The cohort: disjoint from everything reserved, and five records exactly.
# --------------------------------------------------------------------------------------


def test_the_slice_holds_fourteen_scenarios():
    assert len(ROWS) == suite.EXPECTED_ITEMS == 14
    assert len(BY_ID) == 14


def test_every_trajectory_carries_exactly_two_scenarios():
    counts = {name: sum(1 for row in ROWS if row["trajectory"] == name)
              for name in suite.TRAJECTORIES}
    assert counts == {name: 2 for name in suite.TRAJECTORIES}
    assert len(counts) == 7


def test_no_selected_record_is_a_dev_or_holdout_record():
    selected = {int(row["hadm_id"]) for row in ROWS}
    assert selected == set(suite.SELECTED_HADM_IDS)
    assert len(selected) == 5
    assert selected & suite.DEV_HADM_IDS == set()
    assert selected & suite.HOLDOUT_HADM_IDS == set()
    # And nothing the census inventoried as holdout capacity either: those five records
    # exist for the blind suite, and an evaluation slice that consumed them would trade a
    # reusable record for a one-off comparison.
    assert selected & suite.RESERVED_HADM_IDS == set()


def test_the_slice_is_four_inpatients_and_one_outpatient():
    from evaluation.outpatient_oracle import OUTPATIENT_HADM_IDS as ORACLE

    selected = {int(row["hadm_id"]) for row in ROWS}
    outpatient = selected & set(ORACLE)
    assert outpatient == {90000054}, sorted(outpatient)
    assert len(selected - outpatient) == 4


def test_the_archetype_labels_agree_with_the_records():
    for row in ROWS:
        assert row["archetype"] == suite.ARCHETYPES[int(row["hadm_id"])]


def test_seventy_two_census_records_remain_unused_and_sixty_eight_are_free():
    inventory = json.loads(
        (REPO / "evaluation" / "cohort_provenance_inventory.json").read_text())
    census = {int(hadm) for hadm in inventory["all_records"]}
    assert len(census) == 87
    excluded = suite.DEV_HADM_IDS | suite.HOLDOUT_HADM_IDS
    assert len(excluded) == 10
    selected = {int(row["hadm_id"]) for row in ROWS}
    assert not (selected & excluded)
    # 87 - 10 - 5 = 72 records no suite has used.
    assert len(census - excluded - selected) == 72
    # Four of those 72 are the census's own `recommended_holdout` archetypes. The slice
    # leaves them alone as well, so the blind holdout is not built out of whatever survives
    # a selection programme: 68 are untouched by anything.
    assert len(census - excluded - selected - suite.RESERVED_HADM_IDS) == 68


# --------------------------------------------------------------------------------------
# 2. The shape the harness consumes.
# --------------------------------------------------------------------------------------


def test_every_row_validates():
    suite.validate_rows(ROWS)


def test_an_empty_suite_is_refused():
    with pytest.raises(suite.SelectionSliceError):
        suite.validate_rows([])


def test_a_row_that_names_a_dev_record_is_refused():
    import copy

    row = copy.deepcopy(BY_ID["sel_t1_01"])
    row["hadm_id"] = "90000015"
    with pytest.raises(suite.SelectionSliceError):
        suite.validate_rows([row] * 0 + [row])


def test_every_row_parses_into_the_item_model():
    items = suite.selection_slice_items(manifest_provider=_manifest_for)
    assert len(items) == 14
    for item in items:
        assert item.patient_manifest
        assert len(item.prompts) <= MAX_TURNS
        assert item.workflow_type == suite.workflow_for(len(item.prompts))


def test_every_turn_count_matches_its_trajectory():
    for row in ROWS:
        wanted = suite.TURNS_PER_TRAJECTORY[row["trajectory"]]
        assert len(row["turns"]) == wanted, row["id"]


def test_the_loop_back_row_resolves_from_what_the_service_proposed():
    item = DatasetItemInput.model_validate(
        suite.item_payload(BY_ID["sel_t2_01"], dict(STUB_MANIFEST)))
    assert placeholder_turns(item.prompts) == {0: 0}
    proposed = {"follow_ups": ["How does his prior utilisation affect the estimate?"]}
    assert resolve_prompt(item.prompts[1], [proposed]) == proposed["follow_ups"][0]


def test_the_risk_driver_row_declares_its_intent():
    item = DatasetItemInput.model_validate(
        suite.item_payload(BY_ID["sel_t2_02"], dict(STUB_MANIFEST)))
    assert item.declarations[1]["intent"] == "risk_driver"
    # A declared intent is not a placeholder: the wording is the bar's, and it is typed.
    assert placeholder_turns(item.prompts) == {}


def test_every_command_named_is_one_the_agent_has():
    # `/risk` was retired on 2026-10-26. A row naming it would be refused as an unknown
    # command at run time and would measure an error path rather than a capability.
    for row in ROWS:
        command = row.get("command")
        if command is not None:
            assert command in suite.COMMAND_WIRES, row["id"]
        # A free-text cold start has no command at all, and that absence is the point of
        # the trajectory.
        if command is None:
            assert row["trajectory"] == "t5_freetext_cold_start", row["id"]
        for turn in row["turns"]:
            prompt = turn if isinstance(turn, str) else (turn.get("prompt") or "")
            if prompt.startswith("/"):
                wire = prompt[1:].split()[0]
                assert wire in suite.COMMAND_WIRES, f"{row['id']}: /{wire}"


def test_the_slice_exercises_the_literature_channel():
    """`/pubmed` has to appear somewhere, or the comparison never measures it.

    Every candidate model is being judged on whether it can cite a published record it
    actually retrieved, and the join between the prose and the reference drawer is the
    part that broke live on 2026-11-12. A slice with no literature turn would score that
    capability as absent rather than untested.
    """
    literature_rows = [
        row for row in ROWS
        if any(isinstance(turn, str) and turn.startswith("/pubmed")
               for turn in row["turns"])
    ]

    assert literature_rows, "no scenario in the slice runs /pubmed"


def test_the_literature_row_names_the_artefact_it_must_not_produce():
    """A trap without a statement of what it traps scores nothing.

    The forbidden claims on the literature row are the four shapes the remediation run
    reported: a placeholder row, an attribution across the two channels, a record numbered
    with something that is not a page number, and a citation to a record the search never
    returned.
    """
    row = BY_ID["sel_t4_02"]
    claims = " ".join(row["expected_output"]["forbidden_claims"])

    assert "Not Available" in claims
    assert "Clinical Note" in claims
    assert "index that is not a citation number" in claims
    assert "did not return" in claims


# --------------------------------------------------------------------------------------
# 3. The ground truth the judge is handed.
# --------------------------------------------------------------------------------------


def test_every_row_carries_facts_and_a_behaviour():
    for row in ROWS:
        expected = row["expected_output"]
        assert expected["facts"], row["id"]
        assert expected["behaviour"].strip(), row["id"]


def test_the_adversarial_rows_carry_forbidden_claims():
    # The three trajectories that exist to catch a specific wrong answer must say what the
    # wrong answer is; a trap with no statement of what it traps scores nothing.
    for trajectory in ("t6_adversarial_false_premise", "t7_adversarial_boundaries",
                       "t2_dynamic_affordance_loopback"):
        for row in ROWS:
            if row["trajectory"] != trajectory:
                continue
            assert row["expected_output"].get("forbidden_claims"), row["id"]


def test_the_judge_is_given_the_answer_key_for_every_row():
    for row in ROWS:
        block = judge_call._scenario_constraints({}, row["expected_output"])
        assert block, row["id"]
        assert "forbidden_claims" in block or "facts" in block


def test_the_judge_prompt_renders_every_row_without_error():
    for row in ROWS:
        payload = suite.item_payload(row, dict(STUB_MANIFEST))
        output = {
            "turn_responses": ["an answer"] * len(row["turns"]),
            "turn_prompts": [str(turn) if isinstance(turn, str) else "composed"
                             for turn in row["turns"]],
            "turn_sources": [[] for _ in row["turns"]],
            "tool_evidence": [],
        }
        prompt = judge_call.build_user_prompt(
            payload, output, expected_output=row["expected_output"])
        assert "PATIENT MANIFEST" in prompt
        assert "SCENARIO GROUND TRUTH" in prompt


def test_the_rubric_version_is_pinned_on_every_row():
    for row in ROWS:
        assert row["metadata"]["rubric_version"] == "v8"
        assert sync_selection_slice.RUBRIC_VERSION == "v8"


def test_the_metadata_carries_what_a_report_needs():
    for row in ROWS:
        metadata = row["metadata"]
        assert metadata["trajectory"] == suite.TRAJECTORIES[row["trajectory"]]
        assert metadata["archetype"] == row["archetype"]
        assert metadata["command"] == row.get("command")


# --------------------------------------------------------------------------------------
# 4. Zero-LLM provenance.
# --------------------------------------------------------------------------------------

GENERATIVE_SDKS = ("anthropic", "openai", "google.genai", "langchain", "langfuse")

SPRINT_TWO_MODULES = (
    REPO / "evaluation" / "platform" / "selection_slice.py",
    REPO / "evaluation" / "platform" / "sync_selection_slice.py",
    REPO / "evaluation" / "inventory" / "selection_slice_fact_sheet.py",
    Path(__file__).resolve(),
)


def _direct_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def test_no_sprint_two_module_imports_a_generative_sdk():
    offenders = {}
    for path in SPRINT_TWO_MODULES:
        hits = {name for name in _direct_imports(path)
                if any(name == sdk or name.startswith(sdk + ".")
                       for sdk in GENERATIVE_SDKS)}
        if hits:
            offenders[str(path.relative_to(REPO))] = sorted(hits)
    # `langfuse` is the client the sync writes to, not a model. Everything else would be a
    # module that could synthesise a fact instead of reading one. The test module itself
    # imports it only to stand a mock in its place.
    allowed_langfuse = {"evaluation/platform/sync_selection_slice.py",
                        "tests/evaluation/test_selection_slice_schema.py"}
    offenders = {path: hits for path, hits in offenders.items()
                 if not (path in allowed_langfuse and set(hits) == {"langfuse"})}
    assert offenders == {}, offenders


def test_the_scenario_module_imports_no_model_client_at_all():
    imports = _direct_imports(REPO / "evaluation" / "platform" / "selection_slice.py")
    assert imports == {"__future__", "json", "pathlib", "typing",
                       "evaluation.platform.schemas"}, sorted(imports)


def test_the_slice_prompts_are_not_reused_from_the_dev_suite():
    dev = [json.loads(line) for line in
           (REPO / "evaluation" / "datasets" / "dev_suite.jsonl").read_text().splitlines()
           if line.strip()]
    dev_prompts = set()
    for row in dev:
        for turn in row["turns"]:
            dev_prompts.add(turn if isinstance(turn, str) else (turn.get("prompt") or ""))
    ours = set()
    for row in ROWS:
        for turn in row["turns"]:
            ours.add(turn if isinstance(turn, str) else (turn.get("prompt") or ""))
    # A loop-back placeholder is a template the harness fills at run time, not a prompt, so
    # two suites sharing one is not a reuse of wording.
    ours = {prompt for prompt in ours if "{{" not in prompt}
    dev_prompts = {prompt for prompt in dev_prompts if "{{" not in prompt}
    overlap = {prompt for prompt in ours & dev_prompts if len(prompt) > 12}
    assert overlap == set(), overlap


# --------------------------------------------------------------------------------------
# 5. The sync script, end to end against a mock client.
# --------------------------------------------------------------------------------------


class _Row:
    def __init__(self, item_id: str) -> None:
        self.id = item_id


class _Page:
    def __init__(self, rows) -> None:
        self.data = rows


class _DatasetItems:
    def __init__(self, store: dict) -> None:
        self._store = store

    def list(self, dataset_name, limit=100, page=1):
        ids = sorted(self._store)
        start = (page - 1) * limit
        return _Page([_Row(item_id) for item_id in ids[start:start + limit]])

    def delete(self, item_id):
        self._store.pop(item_id, None)


class _Api:
    def __init__(self, store: dict) -> None:
        self.dataset_items = _DatasetItems(store)


class FakeLangfuse:
    """Enough of the client for the sync: dataset, items, paging, flush."""

    created: list[str] = []

    def __init__(self) -> None:
        self.datasets: dict[str, dict] = {}
        self.items: dict[str, dict] = {}
        self.flushed = 0
        self.api = _Api(self.items)

    def create_dataset(self, name, description=None, metadata=None, **kwargs):
        self.datasets[name] = {"description": description, "metadata": metadata}

    def create_dataset_item(self, dataset_name, id, input, expected_output,
                            metadata=None, **kwargs):
        self.items[id] = {"input": input, "expected_output": expected_output,
                          "metadata": metadata}

    def flush(self) -> None:
        self.flushed += 1


@pytest.fixture()
def fake_langfuse(monkeypatch):
    import langfuse

    holder = {}

    def _factory(*args, **kwargs):
        client = FakeLangfuse()
        holder["client"] = client
        return client

    monkeypatch.setattr(langfuse, "Langfuse", _factory)
    monkeypatch.setattr("evaluation.platform.fixtures.manifest_from_bigquery",
                        lambda hadm_ids: _manifest_for(hadm_ids))
    return holder


def test_the_sync_writes_fourteen_items_and_reports_them(fake_langfuse, capsys):
    assert sync_selection_slice.main(["--commit"]) == 0
    client = fake_langfuse["client"]
    assert len(client.items) == 14
    assert set(client.items) == set(BY_ID)
    dataset = client.datasets[sync_selection_slice.DATASET]
    assert dataset["metadata"]["rubric_version"] == "v8"
    assert dataset["metadata"]["purpose"] == "multi-model-selection-slice"
    assert dataset["metadata"]["scenario_count"] == 14
    assert dataset["metadata"]["patient_count"] == 5
    assert client.flushed == 1
    printed = capsys.readouterr().out
    assert "ecc-model-selection-slice-v1: 14 item(s) after upload" in printed


def test_every_synced_item_carries_the_manifest_and_the_answer_key(fake_langfuse):
    sync_selection_slice.main(["--commit"])
    for item_id, item in fake_langfuse["client"].items.items():
        assert item["input"]["patient_manifest"], item_id
        assert item["input"]["expectations"], item_id
        assert item["expected_output"]["facts"], item_id
        assert item["metadata"]["rubric_version"] == "v8"


def test_the_sync_is_idempotent(fake_langfuse):
    sync_selection_slice.main(["--commit"])
    sync_selection_slice.main(["--commit"])
    assert len(fake_langfuse["client"].items) == 14


def test_the_dry_run_writes_nothing(fake_langfuse, capsys):
    assert sync_selection_slice.main([]) == 0
    assert "would write 14 items" in capsys.readouterr().out


def test_the_sync_refuses_a_record_without_a_manifest(fake_langfuse, monkeypatch):
    monkeypatch.setattr("evaluation.platform.fixtures.manifest_from_bigquery",
                        lambda hadm_ids: {})
    with pytest.raises(SystemExit):
        sync_selection_slice.main(["--commit"])
