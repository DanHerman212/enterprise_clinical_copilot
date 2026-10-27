"""TF-10: a follow-up that quotes a previous turn's chart evidence must cite it.

Gate 1 Canary Run 4 scored `dev_t2_03` Turn 2 `citation = 1`. The chip asked for the
`oxycodone SR` taper, which Turn 1's `/ehr` had already retrieved; the ReAct answer
quoted that passage and carried no `^[n]` marker, because the prompt said transcript
evidence needed no citation and the citation layer resolved markers against this
turn's calls alone. Both halves are now gone: `graph._render_tool_result` numbers
passages continuously across the replayed turns and the current one, and
`compose_presentation` resolves against that same concatenation.

These tests pin the contract from the rendering end and from the resolution end: the
number a replayed passage is shown with, and the passage a marker pointing at it
resolves to.
"""

from __future__ import annotations

import json

from services.agent.presentation.a2ui import compose_presentation
from services.agent.graph import _render_tool_result

#: Turn 1's `/ehr` on 90000015, as it is stored on the turn and replayed on Turn 2.
REPLAYED = [{
    "name": "rag_search_sections",
    "args": {"hadm_id": 90000015},
    "response": {
        "returned": 2,
        "query": "discharge medications",
        "passages": [
            {"section": "discharge_medications",
             "text": "oxycodone SR 20 mg q.12h for five days, then 10 mg q.12h "
                     "for five days"},
            {"section": "discharge_instructions",
             "text": "Continue Coumadin; INR every Monday and Thursday."},
        ],
    },
}]

#: What the follow-up turn itself called — a third passage, numbered after the two
#: already on screen.
CURRENT = [{
    "name": "rag_search",
    "args": {"hadm_id": 90000015},
    "response": {
        "returned": 1,
        "query": "hospital course",
        "passages": [{"section": "brief_hospital_course",
                      "text": "Pain was managed by the orthopedic pain service."}],
    },
}]

#: A free-text answer citing across the boundary: `^[1]` and `^[2]` are Turn 1's
#: passages, `^[3]` is this turn's.
ANSWER = (
    "The discharge record tapers oxycodone SR over ten days ^[1], and the "
    "anticoagulant is unchanged ^[2].\n\n"
    "Her post-operative pain was managed by the orthopedic pain service ^[3]."
)


def test_a_replayed_passage_is_shown_with_the_number_the_answer_must_use():
    """The offset is what makes a marker on Turn 2 point at Turn 1's passage.

    Rendering the replayed calls at offset 0 would label the taper `^[1]` on Turn 2
    as well, while the answer's `^[1]` is resolved against both turns' passages.
    """
    rendered = _render_tool_result(REPLAYED[0]["response"], offset=2)

    cites = [passage["cite"] for passage in json.loads(rendered)["passages"]]
    assert cites == [3, 4]
    # And the payload the citation layer resolves against stays unlabelled.
    assert all("cite" not in p for p in REPLAYED[0]["response"]["passages"])


def test_a_follow_up_citing_a_replayed_passage_resolves_to_that_passage():
    out = compose_presentation(
        "What is the dosing and taper schedule for oxycodone SR at discharge?",
        ANSWER, CURRENT, evidence=[*REPLAYED, *CURRENT],
    )

    assert [source["section"] for source in out["sources"]] == [
        "discharge_medications", "discharge_instructions", "brief_hospital_course",
    ]
    assert [source["cite"] for source in out["sources"]] == [1, 2, 3]
    assert "oxycodone SR 20 mg" in out["sources"][0]["text"]
    # A marker that landed on a passage the composer was never shown is the defect
    # this test exists to prevent, and `verified` is where it surfaces.
    assert out["verified"] is True
    assert out["scope"] == "patient_chart"


def test_resolving_against_this_turn_alone_misses_the_replayed_passage():
    """The pre-fix behaviour, kept as the statement of what went wrong.

    `evidence` omitted means the composer sees only the current turn's single
    passage, so `^[1]` clamps to it — the taper marker lands on the hospital course.
    """
    out = compose_presentation(
        "What is the dosing and taper schedule for oxycodone SR at discharge?",
        ANSWER, CURRENT,
    )

    assert out["sources"][0]["section"] == "brief_hospital_course"
    assert "oxycodone SR" not in out["sources"][0]["text"]
