"""Gap 4 — the recall gate measures the candidate, and says so durably.

The defect these tests pin had three parts. The gate was optional, so a
deploy without a report promoted an unmeasured index. The report it consumed
could only describe a state older than the index being promoted, because
nothing measured the candidate. And the verdict was written to a temporary
directory, so no reviewer could reach it.

The load-bearing test here is the join: a run whose manifest names a different
index must not be accepted as the source of the measurement, however recent it
is. Without that check the gate is decorative, because a report about the
previous index authorises the next one just as convincingly.
"""

import pathlib
import re

import pytest

from scripts.agent import recall_gate

REPO = pathlib.Path(__file__).resolve().parents[2]
INDEX_A = ("projects/p/locations/l/indexes/5889384500600766464")
INDEX_B = ("projects/p/locations/l/indexes/2371299135438454784")


class _Artifact:
    def __init__(self, uri):
        self.uri = uri


class _Output:
    def __init__(self, uri):
        self.artifacts = [_Artifact(uri)]


class _Task:
    def __init__(self, name, outputs):
        self.task_name = name
        self.outputs = {key: _Output(uri) for key, uri in outputs.items()}


def _run(name, *, chunks=None, ingest=None, manifest=None):
    """A pipeline run as the API returns one: task details with artifact URIs."""
    tasks = []
    if chunks:
        tasks.append(_Task("chunk-notes", {"chunks": chunks}))
    if ingest:
        tasks.append(_Task("embed-chunks", {"ingest": ingest}))
    if manifest:
        tasks.append(_Task("build-index", {"manifest": manifest}))
    job = type("Job", (), {})()
    job.display_name = name
    job.task_details = tasks
    return job


def _jobs(*runs):
    """A stand-in for PipelineJob.list, which takes filter and order_by."""
    return lambda **_: list(runs)


def test_a_run_that_built_another_index_is_not_accepted():
    """Recency is not provenance: this is the whole point of the join."""
    reader = lambda uri: {"tree_ah_index": INDEX_B}  # noqa: E731
    with pytest.raises(SystemExit) as exc:
        recall_gate.run_for_index(
            INDEX_A,
            jobs=_jobs(_run("rag-ingest-newest", chunks="gs://c", ingest="gs://i",
                            manifest="gs://m2")),
            manifest_reader=reader,
        )

    assert INDEX_A in str(exc.value), "the refusal must name what it could not find"


def test_the_run_that_built_the_index_supplies_the_measurement_inputs():
    reader = lambda uri: {"tree_ah_index": INDEX_A,  # noqa: E731
                          "data_fingerprint": "abc123"}
    source = recall_gate.run_for_index(
        INDEX_A,
        jobs=_jobs(_run("rag-ingest-1", chunks="gs://chunks", ingest="gs://ingest",
                        manifest="gs://manifest")),
        manifest_reader=reader,
    )

    assert source["chunks"] == "gs://chunks"
    assert source["ingest"] == "gs://ingest"
    assert source["data_fingerprint"] == "abc123"
    assert source["run"] == "rag-ingest-1"


def test_a_matching_run_is_found_behind_a_newer_unrelated_one():
    """The newest run is tried first but not trusted by default."""
    seen = []

    def reader(uri):
        seen.append(uri)
        return {"tree_ah_index": INDEX_B if uri == "gs://other" else INDEX_A}

    source = recall_gate.run_for_index(
        INDEX_A,
        jobs=_jobs(
            _run("rag-ingest-newest", chunks="gs://c1", ingest="gs://i1",
                 manifest="gs://other"),
            _run("rag-ingest-older", chunks="gs://c2", ingest="gs://i2",
                 manifest="gs://match"),
        ),
        manifest_reader=reader,
    )

    assert seen == ["gs://other", "gs://match"]
    assert source["chunks"] == "gs://c2"


def test_a_run_without_artifacts_cannot_authorise_a_measurement():
    """A manifest alone is not enough; the job needs the corpus to read."""
    reader = lambda uri: {"tree_ah_index": INDEX_A}  # noqa: E731
    with pytest.raises(SystemExit):
        recall_gate.run_for_index(
            INDEX_A,
            jobs=_jobs(_run("rag-ingest-1", manifest="gs://manifest")),
            manifest_reader=reader,
        )


def _report(recall_at_10: float, empty_rate: float = 0.0) -> dict:
    return {"num_queries": 100, "top_k": 10, "seed": 42,
            "empty_result_rate": empty_rate,
            "recall": {"@1": recall_at_10, "@5": recall_at_10, "@10": recall_at_10},
            "per_query": [{"q": 0, "@1": recall_at_10, "@5": recall_at_10,
                           "@10": recall_at_10}]}


def test_a_report_above_the_threshold_passes():
    passed, _result, failing = recall_gate.judge(
        _report(0.999), corpus="demo", index_name="rag-tree-ah-test")

    assert passed, failing


def test_a_threshold_the_job_did_not_measure_fails_the_gate():
    """A configured maximum with no measurement must not read as satisfied.

    This is how the gate used to behave when it was asked at all: the recall
    job reported no empty-result rate, so a threshold for one failed every
    promotion. The measurement now exists; the fail-closed reading stays.

    What changed on 2026-09-18 is which refusal this is. The missing metric used
    to be reported as a FAILED metric, so a deploy that could not measure the
    empty-result rate printed "refused on ['empty_result_rate']" and sent the
    operator to the corpus — while the corpus was fine and the component doing
    the measuring was in an image four weeks older than the source. It is not a
    breach, it is not a pass: it is its own state, and it is named as such.
    """
    report = _report(0.999)
    del report["empty_result_rate"]

    passed, result, failing = recall_gate.judge(
        report, corpus="demo", index_name="rag-tree-ah-test")

    assert not passed, "nothing measured must not authorise a promotion"
    assert result.unmeasured() == ["empty_result_rate"]
    assert failing == [], "a threshold nobody measured has not been breached"


def test_a_measured_threshold_that_missed_is_the_only_thing_called_failing():
    """The two refusals must stay distinguishable in both directions.

    A report that carries the empty-result rate and breaks the ceiling is a
    verdict on the corpus, and it lands in `failing`. A report that carries no
    rate has nothing to answer with, and it lands in `unmeasured`. Collapsing
    either into the other is how the wrong conclusion got drawn.
    """
    passed, result, failing = recall_gate.judge(
        _report(0.999, empty_rate=0.9), corpus="demo",
        index_name="rag-tree-ah-test")

    assert not passed
    assert failing == ["empty_result_rate"]
    assert result.unmeasured() == []


def test_a_report_with_an_empty_result_rate_above_the_maximum_fails():
    passed, _result, failing = recall_gate.judge(
        _report(0.999, empty_rate=0.9), corpus="demo",
        index_name="rag-tree-ah-test")

    assert not passed
    assert "empty_result_rate" in failing


def test_a_report_below_the_threshold_fails_and_names_the_metric():
    passed, _result, failing = recall_gate.judge(
        _report(0.0), corpus="demo", index_name="rag-tree-ah-test")

    assert not passed
    assert failing, "a failure has to say what failed"


def test_the_deployment_name_carries_the_measurement():
    label = recall_gate.label("rag-tree-ah-20260903001405", _report(0.9924))

    assert label == "rag-tree-ah-20260903001405 recall@10=0.9924"
    assert len(recall_gate.label("x" * 200, _report(0.5))) <= recall_gate.LABEL_MAX


def test_the_evidence_path_is_keyed_by_corpus_and_by_index():
    first = recall_gate.out_dir("demo", "rag-tree-ah-1", "20260918-120000")
    second = recall_gate.out_dir("mimic", "rag-tree-ah-1", "20260918-120000")
    third = recall_gate.out_dir("demo", "rag-tree-ah-2", "20260918-120000")

    assert first != second, "corpora must not share a directory"
    assert first != third, "two indexes must not share a directory"
    assert first.startswith("gs://"), first


def test_the_deploy_cannot_promote_without_measuring():
    """The gate used to be a flag, and the flag was easy to forget.

    Stated over every promotion, because there is now more than one: a promotion
    that measures the candidate and judges it in this run, and a promotion that
    reuses a measurement already on file for an index whose contents and
    thresholds have not changed. Both rest on a measurement; neither rest on a
    flag, a caller's word, or nothing at all.
    """
    source = (REPO / "scripts/agent/deploy_rag.py").read_text()

    assert "--recall-report" not in source, "a supplied report is not a measurement"
    assert "recall_gate.judge(" in source

    promotions = list(re.finditer(r"_deploy\(c, ep_name, index_name, LIVE_ID", source))
    assert promotions, "the promotion path moved; point this at its new wording"
    for promotion in promotions:
        before = source[:promotion.start()]
        fresh = "recall_gate.judge(" in before
        stored = "reused = recall_gate.prior_measurement(" in before
        assert fresh or stored, (
            "a promotion at offset " f"{promotion.start()} rests on no measurement")
        if stored:
            # A reused measurement may not be invented: it has to have passed,
            # to name this index, and to have applied these thresholds.
            assert "prior_measurement" in before
    assert "ROLLED BACK" in source


def test_a_reused_measurement_requires_the_index_and_the_thresholds_to_match():
    """Reuse is a claim about what did not change, so it is checked, not assumed."""
    source = (REPO / "scripts/agent/recall_gate.py").read_text()

    assert "if not payload.get(\"passed\")" in source
    assert "payload.get(\"data_fingerprint\") != data_fingerprint" in source
    assert "get(\"recall_at_10\") != recall_min" in source
    assert "get(\"empty_result_rate\") != empty_max" in source


def test_prior_measurement_ignores_a_report_that_did_not_pass():
    """A stored failure must not authorise anything, and must not stop the
    search from finding an older report that did pass."""
    import json as _json
    from types import SimpleNamespace

    def _entry(passed, fingerprint="fp1", recall=0.99, empty=0.05):
        return _json.dumps({
            "passed": passed, "index_name": "rag-tree-ah-1",
            "data_fingerprint": fingerprint,
            "metrics": {"recall_at_10": recall},
            "thresholds": {"recall_at_10": 0.90},
            "max_thresholds": {"empty_result_rate": empty},
        })

    blobs = {
        "rag/recall/demo/rag-tree-ah-1/20260918-020000/eval_results.json":
            _entry(passed=False),
        "rag/recall/demo/rag-tree-ah-1/20260918-010000/eval_results.json":
            _entry(passed=True),
    }

    class _Blob:
        def __init__(self, text):
            self._text = text

        def download_as_text(self):
            return self._text

    class _Client:
        """Just enough of the storage client for the lookup to run."""

        def bucket(self, name):
            return self

        def blob(self, name):
            return _Blob(blobs[name])

        def list_blobs(self, bucket, prefix=""):
            return [SimpleNamespace(name=name) for name in blobs]

    client = _Client()

    found = recall_gate.prior_measurement(
        "demo", "rag-tree-ah-1", data_fingerprint="fp1",
        recall_min=0.90, empty_max=0.05, client=client)
    assert found, "the passing report must be found"
    assert found["uri"].endswith("20260918-010000/eval_results.json")
    assert found["recall_at_10"] == 0.99

    # Different index contents are not this index's measurement.
    assert recall_gate.prior_measurement(
        "demo", "rag-tree-ah-1", data_fingerprint="fp2",
        recall_min=0.90, empty_max=0.05, client=client) is None

    # Nor is a report judged by thresholds other than the ones in force now.
    assert recall_gate.prior_measurement(
        "demo", "rag-tree-ah-1", data_fingerprint="fp1",
        recall_min=0.999, empty_max=0.05, client=client) is None


def test_the_recall_job_names_nothing_it_could_outlive():
    """Its constants named a deleted endpoint and a month-old pipeline run."""
    source = (REPO / "scripts/agent/submit_recall_job.py").read_text()

    assert "indexEndpoints/" not in source
    assert "pipeline-root/778397675435/" not in source
    assert "recall_gate.run_for_index" in source


def test_the_recall_job_measures_the_empty_result_rate():
    """A threshold the report never carries fails every gate that reads it."""
    from services.mcp.pipelines.recall_k import empty_result_rate

    assert empty_result_rate([]) == 0.0
    assert empty_result_rate([["a"], ["b"]]) == 0.0
    assert empty_result_rate([["a"], []]) == 0.5
    assert empty_result_rate([[], [], [], ["a"]]) == 0.75
    assert "empty_result_rate" in (
        REPO / "services/mcp/pipelines/recall_k.py").read_text()
