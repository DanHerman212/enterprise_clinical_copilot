"""The evaluation case set: the commands, and the follow-up that follows each.

A case is one admission and one command. The first question is the *literal slash
command*, because that is what the product dispatches on: a leading `/` resolves
against the command table deterministically, while prose is matched against a keyword
table that does not contain every command. Sending the prose form of `/ehr` is what
voided the 2026-10-17 run — the reasoning loop answered instead, `/ehr` executed zero
times across 246 turns, and every preflight gate still reported green because nothing
asserted which command ran.

**The follow-up is authored here.** Each one refers back to the first answer with a
demonstrative — *that*, *those* — which has no referent unless the first answer is in
the conversation. That is the property under test: the agent holds nothing between
requests, so a second turn is answerable only if the caller sends the first turn back
with it, and an answer that reads correctly therefore evidences that the earlier turn
was both sent and used.

None of them asks for a decision, so none puts the safety dimension of the rubric at
risk by construction. Each is a question a physician asks next.
"""

from __future__ import annotations

import sys
from pathlib import Path

# The chain's own wording lives in the agent package, which this harness sits
# beside rather than inside. Imported so a change to the product's chips is a
# change to the case set, without a second place to make it.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Suite 1 paths: the four commands every inpatient admission is tested across.
#
# Deliberately four, not five. Suite 1 is 82 admissions x 4 paths = 328 pairs =
# 656 turns, and `/pubmed` is a literature command that takes no patient of its
# own — measured in Suite 2 instead, where its queries are written out.
CENSUS_PROMPTS = ("ehr", "risk", "meds", "labs")

# Every command the run must be shown to execute at least once. Superset of the
# census paths; this is what the coverage assertion and the preflight iterate.
COVERAGE_COMMANDS = CENSUS_PROMPTS + ("pubmed",)

# Kept as the census list so existing callers (collect_http, entrypoint) that count
# Suite 1 cases keep counting four.
PROMPT_NAMES = CENSUS_PROMPTS

# What each prompt actually sends: the literal slash command.
#
# A leading slash is resolved against the command table deterministically; prose is
# matched against a keyword table that does not contain every command. `summarize`
# was the old key for `/ehr` and is retained as an alias so pilot scripts and older
# case files keep working — new code should use `ehr`.
SLASH_COMMANDS = {
    "ehr": "/ehr",
    "risk": "/risk",
    "meds": "/meds",
    "labs": "/labs",
    "pubmed": "/pubmed",
}

PROMPT_ALIASES = {"summarize": "ehr"}

# What the product calls each command on the wire. The harness key and the wire value
# are not always the same name: `/ehr` is carried as `summarize`, because that is the
# internal chain's name for the admission synthesis. Assertions compare harness keys, so
# the mapping happens once, at capture, and a wire value with no entry here still
# surfaces as itself — a command the harness cannot name is a finding, not a pass.
WIRE_COMMAND = {
    "ehr": "summarize",
    "risk": "risk",
    "meds": "meds",
    "labs": "labs",
    # Not "pubmed". Observed on the wire 2026-10-18 in Gate 1: the literature command
    # is carried as `literature`, which is the tool family's own name rather than the
    # slash the reader types. The gate caught the mismatch rather than silently
    # reporting the command as never having run.
    "pubmed": "literature",
}
COMMAND_FROM_WIRE = {wire: key for key, wire in WIRE_COMMAND.items()}

# `/pubmed` is the one command that needs the caller's own words — the literature
# tool takes no admission. Composed as `/pubmed <query>`; the query is part of the
# case, so two literature cases are two different questions.
PUBMED_QUERY = "total knee arthroplasty readmission risk factors"

# The tool each command must reach. Asserted in preflight before a batch is allowed
# to start, so "the command did not run" can never again look like a healthy turn.
COMMAND_TOOLS = {
    "ehr": {"rag_search", "rag_search_sections"},
    "risk": {"predict_readmission"},
    "meds": {"discharge_medications"},
    "labs": {"get_lab_trends"},
    "pubmed": {"search_literature"},
    # alias
    "summarize": {"rag_search", "rag_search_sections"},
}

# The second question of each case. Free text on purpose: turn 2 tests whether a
# follow-up can be answered from the conversation, which is meaningless if it is
# just another command invocation.
FOLLOW_UPS = {
    "ehr": "What was the discharge disposition and follow-up plan?",
    "risk": "Which top clinical drivers contributed most to this risk score?",
    "meds": ("Are any high-risk or anticoagulant medications listed on "
             "discharge?"),
    "labs": ("Which of those laboratory values was most abnormal at "
             "discharge?"),
    "pubmed": "Which of those findings applies to this admission?",
    # alias
    "summarize": "What was the discharge disposition and follow-up plan?",
}


def resolve(prompt: str) -> str:
    """The canonical command name for a prompt key, following aliases."""
    return PROMPT_ALIASES.get(prompt, prompt)


def question_for(prompt: str, hadm_id: int) -> str:
    """The first question: the literal slash command.

    `hadm_id` is not embedded in the text because it travels as its own request
    field. It stays in the signature so a caller holding the admission cannot
    quietly stop passing it.
    """
    del hadm_id
    command = resolve(prompt)
    if command == "pubmed":
        return f"{SLASH_COMMANDS[command]} {PUBMED_QUERY}"
    return SLASH_COMMANDS[command]


def follow_up_for(prompt: str) -> str:
    """The second question of the case, which is not addressed to an admission.

    Deliberately without the admission suffix the first question carries: the
    follow-up refers to the conversation, and naming the admission again would let
    it be answered from the admission alone, which is the thing being tested.
    """
    return FOLLOW_UPS[resolve(prompt)]
