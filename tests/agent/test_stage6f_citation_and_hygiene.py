"""Stage 6F: no Dev-set patient literal in a prompt, and one passage per sentence.

Two things were wrong with the prompt after Stage 6C.

It carried the Dev set into production prose. Every rule written to repair a failing
scenario was illustrated with that scenario's own words — a named antiplatelet, a named
anticoagulant, its discontinuation date, a hemoglobin trajectory with real values, a
reconciliation phrase — so a clinician reading a summary was shown one specific record's
drugs as if they were examples of the rule. A prompt is read by a model about to write
about a DIFFERENT patient, and an example drawn from one record is a template it can
follow into another.

And it said two opposite things about markers. One bullet required a sentence drawing on
two sections to carry both markers; another required every cited passage to state something
the sentence says; the hub allowed a claim "at most one passage" while the chart rules asked
for two on one sentence. A model told both does what the nearer instruction says, and the
answer to `dev_t2_02` cited a second passage on a sentence whose only claim the first
passage stated. The rule is now one: one sentence, one passage, one marker.
"""

from __future__ import annotations

import pytest

from services.agent.engine import prompts
from services.agent.postprocess import citations

#: Every constant that reaches a model. `SYSTEM_PROMPT` is the composed alias and is
#: covered by the constituents; the formatters are separate prompts for separate turns.
PROMPT_CONSTANTS = (
    "CORE_HUB_PROMPT", "CHART_SPOKE", "MEDS_SPOKE", "LABS_SPOKE", "RISK_SPOKE",
    "LITERATURE_SPOKE", "EHR_FORMATTING", "LABS_FORMATTING", "MEDS_FORMATTING",
    "LITERATURE_FORMATTING", "FORMATTER_PREAMBLE", "SYSTEM_PROMPT",
)

#: The Dev set's own wording. A drug name alone is clinical vocabulary; these are the
#: specific literals this repository's prompt carried from a failing scenario's note.
DEV_SET_LITERALS = (
    "Plavix",
    "Lovenox",
    "on no meds",
    "14.0 to 13.4",
    "08/30/2007",
    "09/21/2007",
    "08/21/2007",
    "04/04/2007",
    "dev_t",
    "holdout_",
)


def _prompt_text(name: str) -> str:
    value = getattr(prompts, name)
    assert isinstance(value, str), f"{name} is not a prompt string"
    return value


def test_no_prompt_names_a_dev_set_literal():
    """The rule a failing scenario taught, stated without the scenario's own record.

    A named drug or a note's date in a prompt is not an example of the rule — it is that
    record's data, handed to a model writing about another patient.
    """
    leaks: list[str] = []
    for name in PROMPT_CONSTANTS:
        text = _prompt_text(name)
        for literal in DEV_SET_LITERALS:
            if literal in text:
                leaks.append(f"{name} contains {literal!r}")

    assert not leaks, "; ".join(leaks)


def test_the_chart_rules_keep_one_passage_per_sentence():
    text = " ".join(prompts.EHR_FORMATTING.split())

    assert "ONE PASSAGE PER SENTENCE" in text
    assert "Every cited sentence carries exactly one `^[k]`" in text
    assert "100% of that sentence's clinical claims" in text
    # The sections the measured failure merged, named as separate passages.
    assert ("`history_of_present_illness`, `chief_complaint` and `brief_hospital_course` "
            "are separate passages") in text
    assert "Never merge a finding from `brief_hospital_course` into a" in text


def test_a_note_without_a_history_passage_still_has_a_source_for_the_presentation():
    """The strict rule needs a fallback, or a sparse note has no legal sentence.

    A note may arrive with only `brief_hospital_course`, or as one unsectioned
    `discharge_summary`. The rule that the presentation comes from
    `history_of_present_illness` cannot apply to either, and a rule that cannot apply is a
    rule the model satisfies by reaching for the nearest passage — the diagnosis list.
    """
    text = " ".join(prompts.EHR_FORMATTING.split())

    assert "Only when `history_of_present_illness` is absent" in text
    assert "the one passage that states those symptoms" in text
    assert "never from the diagnosis passage" in text
    # The no-merge rule survives the fallback.
    assert "no two of them are ever merged into one sentence" in text
    assert "never split across two" in text


def test_the_hub_keeps_one_passage_per_sentence():
    text = " ".join(prompts.CORE_HUB_PROMPT.split())

    assert "ONE PASSAGE PER SENTENCE" in text
    assert "A sentence states what ONE `<evidence_items>` passage states" in text
    assert "never attach two numbered markers" in text
    # And the two kinds of evidence that carry no numbered marker are named with it.
    assert "`<literature>` records keep `^[PMID: <id>]`" in text
    assert "`<admission_metadata>` carry no `^[k]`" in text


@pytest.mark.parametrize("wording", [
    "carries both markers",
    "cite both passages on the one sentence",
    "cite every passage the sentence draws on",
    "Two markers are two claims",
    "Give an item its own marker only when it rests on a",
])
def test_no_prompt_still_invites_two_markers_on_one_sentence(wording):
    """Asserted as absences, so the contradictory instruction cannot come back.

    Each of these told the model to put two numbered markers on one sentence. One of them
    was in the hub and one in the chart rules, so a turn could be given both.
    """
    for name in PROMPT_CONSTANTS:
        assert wording not in _prompt_text(name), f"{name} still says {wording!r}"


def test_the_medication_rules_keep_the_two_facts_in_two_sentences():
    text = " ".join(prompts.MEDS_SPOKE.split())

    assert "TWO SEPARATE SENTENCES" in text
    assert "what the dedicated section held" in text
    assert "carrying only that passage's marker" in text


# --- the parser keeps its defence in depth ------------------------------------------

def test_a_cluster_is_still_normalised_and_renumbered():
    """The prompt now forbids two markers on a sentence; the parser still reads them.

    A deterministic narrative (`/meds`, `/labs`) writes its own markers, and a model that
    ignores the rule still produces text the pipeline must not corrupt. So the code that
    expands `^[2], [5]` stays, even though nothing should now ask for it.
    """
    written = "The rates were 4.4% and 1.1% ^[1], [4]."

    assert citations.normalise_carets(written) == "The rates were 4.4% and 1.1% ^[1], ^[4]."

    prose, display = citations.compact_display_numbers(citations.normalise_carets(written))
    assert prose == "The rates were 4.4% and 1.1% ^[1], ^[2]."
    assert display == {"1": 1, "2": 4}


def test_a_cluster_survives_renumber_citations():
    """`renumber_citations` is the second reader, and it sees the same two markers.

    Numbers are preserved and only the shape changes, so the two markers stay the two
    positions the model wrote.
    """
    assert citations.renumber_citations("Findings ^[2], [5].") == "Findings ^[2] ^[5]."
