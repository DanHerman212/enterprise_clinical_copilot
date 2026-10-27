"""Two invariants, both of them global rather than per-surface.

1. Every source a turn publishes produces its reference row, under the same number. The
   two lists are the same list told twice — the inline pills and the drawer rows — so a
   turn cannot draw `1..K` pills and list fewer than `K` rows at the foot of the answer.

2. A command owns one domain. `/pubmed` searches the literature and does nothing else, so
   it cannot append a patient's readmission risk to a literature answer, and the chips it
   offers are about the evidence rather than about the chart.

Both were reported from `/pubmed bilateral lower extremity cellulitis` on 2026-10-26: three
numbered pills and no `REFERENCES` drawer, a `23.0%` risk paragraph nobody asked for, and a
"why does prior inpatient days increase this patient's readmission risk?" chip under an
answer about published evidence.
"""

from __future__ import annotations

from services.agent import followups
from services.agent import routing
from services.agent.a2ui import compose_presentation

SECTIONS = {
    "query": "sections",
    "returned": 2,
    "passages": [
        {"id": "p1", "section": "discharge_diagnosis",
         "text": "Discharge Diagnosis: Acute decompensated heart failure"},
        {"id": "p2", "section": "discharge_instructions",
         "text": "Discharge Instructions: Weigh daily. Follow up in one week."},
    ],
}

SECTIONS_CALL = {"name": "rag_search_sections", "args": {"hadm_id": 90000023},
                 "response": SECTIONS}

MEDS_CALL = {
    "name": "discharge_medications",
    "args": {"hadm_id": 90000023},
    "response": {
        "query": "discharge medications", "returned": 1,
        "passages": [{"id": "m1", "section": "discharge_medications",
                      "text": "Discharge Medications:\nwarfarin 4 mg QD"}],
    },
}

LABS_CALL = {
    "name": "get_lab_trends", "args": {"hadm_id": 90000023},
    "response": {
        "hadm_id": 90000023, "synthetic": True, "note": "Values are synthetic.",
        "trends": [{"key": "creatinine", "label": "Creatinine", "unit": "mg/dL",
                    "reference_low": 0.5, "reference_high": 1.1, "decimals": 2,
                    "series": [{"day": 0.0, "value": 1.05, "flag": "normal"},
                               {"day": 4.0, "value": 1.62, "flag": "high"}]}],
    },
}

PREDICT_CALL = {"name": "predict_readmission", "args": {"hadm_id": 90000023},
                "response": {"hadm_id": 90000023, "probability": 0.23,
                             "threshold": 0.11, "decision": 1,
                             "model_version": "test-v1",
                             "top_factors": [{"feature": "prior_inpatient_days",
                                              "contribution": 0.4,
                                              "direction": "increases"}]}}

ARTICLES = [
    {"pmid": "29950074", "title": "Readmission after lumbar spine surgery.",
     "journal": "World Neurosurgery", "pub_date": "2024 Jul 01",
     "authors": "Proal JD, Moon AS, LaHaise K, et al.",
     "abstract": "BACKGROUND: Readmission is common. RESULTS: The 30-day rate was 4.1%.",
     "url": "https://pubmed.ncbi.nlm.nih.gov/29950074/"},
    {"pmid": "31234567", "title": "Cellulitis management in the emergency department.",
     "journal": "American Family Physician", "pub_date": "2026 Jan 15",
     "authors": "Smith JA, Doe B.",
     "abstract": "BACKGROUND: Cellulitis is common. RESULTS: Outpatient therapy sufficed.",
     "url": "https://pubmed.ncbi.nlm.nih.gov/31234567/"},
]

SEARCH_CALL = {"name": "search_literature",
               "args": {"query": "bilateral lower extremity cellulitis"},
               "response": {"query": "bilateral lower extremity cellulitis",
                            "returned": 2, "articles": ARTICLES}}


def _pairs(presentation: dict) -> tuple[list[int], list[int]]:
    return ([source["cite"] for source in presentation["sources"]],
            [row["index"] for row in presentation["references"]])


# --- invariant 1: sources == references ---------------------------------------

class TestEverySourceHasARow:
    def test_a_note_turn_lists_what_it_cites(self):
        out = compose_presentation(
            "/ehr", "Heart failure.^[1] Follow up in a week.^[2]",
            [SECTIONS_CALL, PREDICT_CALL], command="summarize")

        cites, rows = _pairs(out)
        assert cites == rows
        assert cites[:2] == [1, 2], "the two cited note passages"
        # The comparison table's cohort is a source too, numbered after them, and it has
        # its row — that is what `references` being a view of `sources` means.
        assert out["sources"][2]["kind"] == "Literature"
        assert out["references"][2]["pmid"] == out["sources"][2]["pmid"]

    def test_a_medication_turn_lists_its_passage(self):
        out = compose_presentation(
            "/meds", "Warfarin 4 mg daily.^[1]", [MEDS_CALL], command="meds")

        cites, rows = _pairs(out)
        assert cites == rows == [1]

    def test_a_lab_turn_lists_its_trajectory(self):
        out = compose_presentation(
            "How did the creatinine move?", "It rose to 1.62.^[1]", [LABS_CALL])

        cites, rows = _pairs(out)
        assert cites == rows == [1]
        assert out["references"][0]["title"] == "Creatinine trajectory"

    def test_a_literature_turn_lists_the_records_it_pills(self):
        """The reported defect: three pills, no drawer.

        A literature turn's records were published in `sources` and drawn as pills, but
        `references` was assembled by a different builder that only knew about note
        passages and the comparison table's cohorts.
        """
        out = compose_presentation(
            "/pubmed bilateral lower extremity cellulitis",
            "Two records.^[PMID: 29950074] Another.^[PMID: 31234567]",
            [SEARCH_CALL], command="literature")

        cites, rows = _pairs(out)
        assert cites == rows == [1, 2]
        titles = [row["title"] for row in out["references"]]
        assert titles == [article["title"] for article in ARTICLES]
        # Each row is openable: a citation line, the passage, and the identifier.
        for row in out["references"]:
            assert row["subtitle"].endswith(f"PMID {row['pmid']}")
            assert row["excerpt"]
            assert row["url"].endswith(f"{row['pmid']}/")

    def test_a_turn_with_nothing_to_cite_publishes_nothing_either_way(self):
        out = compose_presentation("what is the risk?", "Cannot say.", [])

        assert out["sources"] == []
        assert out["references"] == []

    def test_every_row_names_itself(self):
        """A row without a title is a number with nothing behind it."""
        for out in (
            compose_presentation("/ehr", "Heart failure.^[1]", [SECTIONS_CALL],
                                 command="summarize"),
            compose_presentation("/meds", "Warfarin.^[1]", [MEDS_CALL], command="meds"),
            compose_presentation("Creatinine?", "It rose.^[1]", [LABS_CALL]),
            compose_presentation("/pubmed q", "Records.^[PMID: 29950074]",
                                 [SEARCH_CALL], command="literature"),
        ):
            for row in out["references"]:
                assert row["title"].strip(), row
                assert row["label"].startswith(f"{row['index']} · "), row


# --- invariant 2: /pubmed owns the literature ---------------------------------

class TestPubmedOwnsOneDomain:
    def test_the_command_declares_no_second_call(self):
        """It used to run the estimate beside the search, so the turn held this
        patient's score while answering a question about published evidence."""
        command = routing.command_for("literature")

        assert command.supplementary == ()
        assert routing.supplementary_arguments(command, 90000023, "cellulitis") == []
        assert routing.supplementary_arguments(command, None) == []

    def test_the_estimate_is_not_even_reachable_on_this_command(self):
        command = routing.command_for("literature")

        assert "predict_readmission" in command.forbidden

    def test_a_literature_turn_is_badged_general_evidence(self):
        out = compose_presentation(
            "/pubmed bilateral lower extremity cellulitis",
            "Two records.^[PMID: 29950074]", [SEARCH_CALL], command="literature")

        assert out["scope"] == "general_evidence"
        assert out["verified"] is False, (
            "published records are somebody else's patients")

    def test_the_turn_draws_no_risk_surface(self):
        """No estimate, no drivers, no cohort table: the question was about the
        literature."""
        out = compose_presentation(
            "/pubmed bilateral lower extremity cellulitis",
            "Two records.^[PMID: 29950074]", [SEARCH_CALL], command="literature")

        types = [c["component"]
                 for c in out["a2ui"]["messages"][1]["updateComponents"]["components"]]
        assert "RiskInfluencers" not in types
        assert "ComparativeRiskMatrix" not in types

    def test_the_chips_offered_are_about_the_evidence(self):
        """The reported chip asked why prior inpatient days raise this patient's risk —
        under an answer about cellulitis."""
        state = {"tool_calls": [SEARCH_CALL, PREDICT_CALL], "replayed_tool_calls": []}

        chips = followups.suggest(
            "/pubmed bilateral lower extremity cellulitis",
            "Two records were retrieved.", scope="general_evidence",
            state=state, literature=True)

        assert len(chips) == 1, chips
        assert "cellulitis" in chips[0]
        # Nothing about this patient got in.
        assert "readmission risk" not in chips[0].lower()
        assert "prior inpatient" not in chips[0].lower()

    def test_a_literature_turn_with_no_records_offers_nothing(self):
        state = {"tool_calls": [{"name": "search_literature", "args": {"query": "q"},
                                 "response": {"query": "q", "returned": 0,
                                              "articles": []}}],
                 "replayed_tool_calls": []}

        assert followups.suggest("/pubmed q", "Nothing was found.",
                                 scope="general_evidence", state=state,
                                 literature=True) == []

    def test_the_chip_asks_about_the_search_query_not_the_wrapper_sentence(self):
        """Reported live: the chip read

          "What else does the published literature report on Search PubMed for recent
           literature on hyperglycemia. For admission 90000067?"

        The console expands a typed `/pubmed hyperglycemia` into that sentence before the
        service sees it, and the chip slotted the expansion and the admission id into its
        template. The search already holds the words, so they come from there.
        """
        state = {
            "tool_calls": [{"name": "search_literature",
                            "args": {"query": "hyperglycemia"},
                            "response": {"query": "hyperglycemia", "returned": 3,
                                         "articles": ARTICLES}}],
            "replayed_tool_calls": [],
        }

        chips = followups.suggest(
            "Search PubMed for recent literature on hyperglycemia. For admission 90000067.",
            "Three records were retrieved.", scope="general_evidence",
            state=state, literature=True)

        assert chips == [
            "What else does the published literature report on hyperglycemia?"]

    def test_the_wrapper_is_stripped_even_with_no_search_in_the_state(self):
        """The fallback has to stand on its own: a turn replayed from the store has no
        tool calls, and the question it was asked is still the wrapped one."""
        chips = followups.suggest(
            "Search PubMed for recent literature on hyperglycemia. For admission 90000067.",
            "Three records were retrieved.", scope="general_evidence",
            state={"tool_calls": [{"name": "search_literature",
                                   "args": {"query": ""},
                                   "response": {"query": "", "returned": 3,
                                                "articles": ARTICLES}}]},
            literature=True)

        assert chips == [
            "What else does the published literature report on hyperglycemia?"]

    def test_a_chart_turn_still_gets_chart_chips(self):
        """The flag must not leak: `/ehr` is the turn that owns the chart, and its
        chips are still drawn from it."""
        state = {"tool_calls": [SECTIONS_CALL], "replayed_tool_calls": []}

        chips = followups.suggest("/ehr", "Heart failure, follow up in a week.",
                                  scope="patient_chart", state=state,
                                  literature=False)

        assert chips, "the chart turn lost every chip it had"
