"""Stage 9B: four fixes, each measured on the turn that produced it.

The audit of 2026-11-11 traced three failures to three different layers, and this file
pins one turn per fix, verbatim from the run that failed.

`dev_t1_06` wrote `^[2] ^[1] ^[3] ^[2]` on a `/pubmed` turn. Settlement compacted the page
to first-appearance order, so the reader was served `^[1] ^[2] ^[3] ^[1]` while the drawer
still numbered the records `1..N` in retrieval order — the pill and the row disagreed by
exactly the swap. `/pubmed` now keeps the numbering its records were published under.

`dev_t2_05` wrote `^[predict_readmission]` twice, a tool name where a passage number
belongs. The vocabulary sweep deleted both and left the space in front of each full stop:
`contribution of 0.0824 .`. The whitespace now goes with the marker.

`dev_t3_02` was withheld by the provider at `HARM_CATEGORY_DANGEROUS_CONTENT`,
`probability: MEDIUM`, `severity: HARM_SEVERITY_LOW` — a discharge medication taper, rated
low severity and blocked for the category a discharge note is written in.

And `dev_t2_02`'s mis-graded sentence was a plan reported as a completed admission:
"he was to be admitted for the procedure" written as "he was admitted for the procedure".
"""

from __future__ import annotations

from services.agent.engine import prompts
from services.agent.postprocess import answer as answer_text
from services.agent.postprocess import citations
from services.agent.presentation.a2ui import compose_presentation
from services.mcp.config import GEMINI_SAFETY_THRESHOLDS

#: `dev_t1_06` turn 1, the raw generation from trace `e1aae61ae86d610455ffa091209d8b5d`
#: before anything settled it.
T1_06_RAW = (
    "Recent evidence suggests that the optimal duration of dual antiplatelet therapy "
    "(DAPT) after coronary stenting depends on balancing ischemic and bleeding risks, "
    "with shorter durations increasingly explored for specific patient populations "
    "^[2]. While one trial found that 1 to 2 months of DAPT followed by clopidogrel "
    "monotherapy failed to demonstrate noninferiority to 12 months of DAPT in acute "
    "coronary syndrome patients due to a numerical increase in cardiovascular events "
    "^[1], another study demonstrated that P2Y12-inhibitor monotherapy after 1 month of "
    "DAPT is noninferior to continued DAPT in low-risk patients with acute myocardial "
    "infarction, while significantly reducing bleeding events ^[3]. Overall, prolonged "
    "DAPT beyond 12 months may reduce stent thrombosis and major adverse cardiovascular "
    "events but consistently increases the risk of bleeding ^[2]."
)

#: The three records that search returned, in the order the model was given them.
T1_06_ARTICLES = [
    {"pmid": "35234821", "title": "Comparison of Clopidogrel Monotherapy After 1 to 2 "
                                 "Months of Dual Antiplatelet Therapy With 12 Months.",
     "journal": "JAMA cardiology", "journal_abbrev": "JAMA Cardiol",
     "pub_date": "2022 Apr 01", "authors": "Watanabe H, et al.",
     "abstract": "Noninferiority of short DAPT was not shown."},
    {"pmid": "37511955", "title": "P2Y12 inhibitor monotherapy after short DAPT.",
     "journal": "Life (Basel, Switzerland)", "journal_abbrev": "Life (Basel)",
     "pub_date": "2023 Jul 01", "authors": "Rossi A, et al.",
     "abstract": "Monotherapy was noninferior and reduced bleeding."},
    {"pmid": "40888726", "title": "Duration of dual antiplatelet therapy after stenting.",
     "journal": "The New England journal of medicine", "journal_abbrev": "N Engl J Med",
     "pub_date": "2025 Sep 01", "authors": "Chen L, et al.",
     "abstract": "Prolonged DAPT reduced thrombotic events and increased bleeding."},
]


def _literature_call() -> dict:
    return {"name": "search_literature",
            "args": {"query": "dual antiplatelet therapy after coronary stenting"},
            "response": {"query": "dual antiplatelet therapy after coronary stenting",
                         "returned": 3, "articles": T1_06_ARTICLES}}


# --------------------------------------------------------------------------------------
# Fix 1: `/pubmed` keeps the numbering its records were published under.
# --------------------------------------------------------------------------------------


def test_a_pubmed_answer_is_not_compacted():
    """The markers stay as written: `2, 1, 3, 2`."""
    settled = answer_text.settle_answer(
        prose=T1_06_RAW,
        deterministic=None,
        tool_calls=[_literature_call()],
        replayed_tool_calls=None,
        command="literature",
    )

    assert settled.prose.count("^[2]") == 2
    assert settled.prose.count("^[1]") == 1
    assert settled.prose.count("^[3]") == 1
    assert settled.prose == T1_06_RAW
    assert settled.display == {}, (
        "a literature answer publishes no map, because it renumbers nothing")


def test_the_pills_and_the_records_name_the_same_paper():
    """The whole point: every pill opens the record the model was shown at that index."""
    settled = answer_text.settle_answer(
        prose=T1_06_RAW,
        deterministic=None,
        tool_calls=[_literature_call()],
        replayed_tool_calls=None,
        command="literature",
    )

    out = compose_presentation(
        "Search PubMed for recent literature on dual antiplatelet therapy after "
        "coronary stenting. For admission 90000023.",
        settled.prose,
        [_literature_call()],
        command="literature",
        citation_map=settled.display,
    )

    assert out["answer"] == T1_06_RAW
    # No compaction happened, so the only map is the identity the presentation states for
    # the numbers the prose actually cites.
    assert out["citation_map"] == {"1": 1, "2": 2, "3": 3}

    sources = out["sources"]
    assert [source["cite"] for source in sources] == [1, 2, 3]
    assert "JAMA" in sources[0]["label"]
    assert "Life" in sources[1]["label"]
    assert "England" in sources[2]["label"]

    # The narrative's pills carry those indices, and the prose's markers are the ones the
    # model wrote — `2, 1, 3, 2` — so pill 2 and drawer row 2 are the same paper.
    narrative = next(component for message in out["a2ui"]["messages"]
                     for component in (message.get("updateComponents") or {})
                     .get("components") or []
                     if component.get("component") == "CitedNarrative")
    assert narrative["text"] == T1_06_RAW
    assert [citation["index"] for citation in narrative["citations"]] == [1, 2, 3]
    assert "Life" in narrative["citations"][1]["label"]

    # And the counts hold: two pills naming record 2, one naming 1, one naming 3.
    assert settled.prose.count("^[2]") == 2
    assert settled.prose.count("^[1]") == 1
    assert settled.prose.count("^[3]") == 1


def test_the_same_answer_on_a_chart_command_is_still_compacted():
    """The exemption is the literature route, not the presence of markers."""
    settled = answer_text.settle_answer(
        prose="A^[2] and B^[1]",
        deterministic=None,
        tool_calls=[{"name": "rag_search", "args": {}, "response": {
            "hadm_id": 90000023, "query": "notes",
            "passages": [{"id": "p1", "section": "brief_hospital_course", "text": "a"},
                         {"id": "p2", "section": "discharge_diagnosis", "text": "b"}],
        }}],
        replayed_tool_calls=None,
        command=None,
    )

    assert settled.prose == "A^[1] and B^[2]"
    assert settled.display == {"1": 2, "2": 1}


# --------------------------------------------------------------------------------------
# Fix 2: the sweep takes the whitespace with the marker.
# --------------------------------------------------------------------------------------


#: `dev_t2_05` turn 2, the raw generation from trace `79bdf6e96a6f29b35ae132f25c590c67`.
T2_05_RAW = (
    "The readmission risk model identifies the \"lowest monocyte count\" as a driver that "
    "increases the patient's risk, with a contribution of 0.0824 ^[predict_readmission]. "
    "The value for this feature is recorded as null in the model's input set "
    "^[predict_readmission]. The patient's clinical notes do not document a monocyte "
    "count, and they do not account for the mechanism by which this factor influences the "
    "readmission risk estimate."
)


def test_a_tool_name_marker_leaves_no_residue():
    prose = citations.drop_unknown_markers(T2_05_RAW)

    assert "predict_readmission" not in prose
    assert " ." not in prose
    assert "0.0824." in prose
    assert "input set." in prose
    assert prose == (
        "The readmission risk model identifies the \"lowest monocyte count\" as a driver "
        "that increases the patient's risk, with a contribution of 0.0824. The value for "
        "this feature is recorded as null in the model's input set. The patient's clinical "
        "notes do not document a monocyte count, and they do not account for the mechanism "
        "by which this factor influences the readmission risk estimate."
    )


def test_the_whole_turn_settles_without_the_residue():
    settled = answer_text.settle_answer(
        prose=T2_05_RAW,
        deterministic=None,
        tool_calls=[],
        replayed_tool_calls=None,
        command=None,
    )

    assert " ." not in settled.prose
    assert "predict_readmission" not in settled.prose
    assert "0.0824." in settled.prose


def test_clean_prose_is_returned_untouched():
    """The reason the tidy runs only when something was removed.

    A whitespace rule that rewrites passing prose is how the meds guard once damaged a
    correct answer, so this asserts identity on text with nothing to remove.
    """
    for prose in (
        "The rates were 4.1% and 3.7%.^[PMID: 35985540, PMID: 33534292]",
        "Stenosis ^[1]. Left ventricular function was preserved ^[2].",
        "No marker here at all.",
        "",
    ):
        assert citations.drop_unknown_markers(prose) is prose


def test_a_section_name_marker_does_not_leave_a_space_before_its_stop():
    assert citations.drop_unknown_markers(
        "Stenosis ^[brief_hospital_course].") == "Stenosis."
    assert citations.drop_unknown_markers(
        "Creatinine ^[tool_x] and eGFR rose.") == "Creatinine and eGFR rose."


# --------------------------------------------------------------------------------------
# Fix 3: the dangerous-content threshold.
# --------------------------------------------------------------------------------------


def test_the_dangerous_content_threshold_is_the_high_band_only():
    """`dev_t3_02` was blocked at `MEDIUM` with `HARM_SEVERITY_LOW`.

    The other three categories are unchanged, and this test names them so a later edit
    cannot move one by accident.
    """
    assert GEMINI_SAFETY_THRESHOLDS["HARM_CATEGORY_DANGEROUS_CONTENT"] == "BLOCK_ONLY_HIGH"
    assert GEMINI_SAFETY_THRESHOLDS["HARM_CATEGORY_HATE_SPEECH"] == \
        "BLOCK_MEDIUM_AND_ABOVE"
    assert GEMINI_SAFETY_THRESHOLDS["HARM_CATEGORY_HARASSMENT"] == "BLOCK_MEDIUM_AND_ABOVE"
    assert GEMINI_SAFETY_THRESHOLDS["HARM_CATEGORY_SEXUALLY_EXPLICIT"] == \
        "BLOCK_MEDIUM_AND_ABOVE"


# --------------------------------------------------------------------------------------
# Fix 4: the prompt rules.
# --------------------------------------------------------------------------------------


def test_the_hub_names_the_marker_bodies_that_are_not_markers():
    """`^[predict_readmission]` is not a citation, and the prompt now says so."""
    hub = " ".join(prompts.CORE_HUB_PROMPT.split())

    assert "A body is a number or a `^[PMID: …]` badge; a tool or tag name is deleted" in hub
    # The tag facts it depends on are still stated.
    assert "`<admission_metadata>` carry no `^[k]`" in hub


def test_the_ehr_prompt_keeps_a_planned_event_prospective():
    text = " ".join(prompts.EHR_FORMATTING.split())

    assert "KEEP THE PASSAGE'S OWN TENSE" in text
    assert "report it as planned or scheduled" in text
    assert ('"he was to be admitted for the procedure" is not "he was admitted for the '
            'procedure"') in text


def test_the_prompts_stay_inside_their_word_budgets():
    assert len(prompts.CORE_HUB_PROMPT.split()) <= 1500
    for name in ("CHART_SPOKE", "MEDS_SPOKE", "LABS_SPOKE", "RISK_SPOKE",
                 "LITERATURE_SPOKE"):
        assert len(getattr(prompts, name).split()) <= 650, name


def test_the_new_prompt_text_carries_no_dev_set_literal():
    """The rule this stage added, stated without the record that taught it."""
    from tests.agent.test_stage6f_citation_and_hygiene import DEV_SET_LITERALS

    for name in ("CORE_HUB_PROMPT", "EHR_FORMATTING"):
        text = getattr(prompts, name)
        for literal in DEV_SET_LITERALS:
            assert literal not in text, f"{name} contains {literal!r}"
