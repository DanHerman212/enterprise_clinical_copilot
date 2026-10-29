"""The evaluation case set: which commands are measured, and what each one sends.

Moved out of the retired collector (`evaluation/agent/prompt_set.py`, 2026-10-29) because
the live platform reads it: `fixtures.py` sizes every census dataset from
`CENSUS_PROMPTS`, and the census builder names its cases from the same tuple. Two copies
of that tuple is how a dataset ends up sized for four commands and seeded with three.

The tool expectations are **derived** from `services.agent.routing.COMMANDS` rather than
retyped (charter Law 4). The old copy in `prompt_set.py` had already drifted: it described
`/ehr` as `{rag_search, rag_search_sections, predict_readmission}` by hand, and nothing
compared it with the command table it was supposed to mirror.

The harness key and the wire value are not always the same name — `/ehr` is carried as
`summarize`, `/pubmed` as `literature` — so the mapping is stated once here and a wire
value with no entry surfaces as itself, which is a finding rather than a pass.
"""

from __future__ import annotations

from services.agent.routing import routing as agent_routing

#: The four commands every inpatient admission is tested across. `/risk` was the fourth
#: until the command was retired on 2026-10-26; `/pubmed` takes its place, because the
#: comparative matrix pins this patient's own estimate above the published cohorts.
CENSUS_PROMPTS: tuple[str, ...] = ("ehr", "pubmed", "meds", "labs")

#: Every command the run must be shown to execute at least once.
COVERAGE_COMMANDS: tuple[str, ...] = CENSUS_PROMPTS

#: What each prompt actually sends: the literal slash command. A leading slash resolves
#: against the command table deterministically; prose is matched against a keyword table
#: that does not contain every command, which is how the 2026-10-17 run had `/ehr`
#: execute zero times while every preflight gate read green.
SLASH_COMMANDS: dict[str, str] = {
    "ehr": "/ehr",
    "meds": "/meds",
    "labs": "/labs",
    "pubmed": "/pubmed",
}

#: The harness key → the name the product reports on the wire.
WIRE_COMMAND: dict[str, str] = {
    "ehr": "summarize",
    "meds": "meds",
    "labs": "labs",
    "pubmed": "literature",
}
COMMAND_FROM_WIRE: dict[str, str] = {wire: key for key, wire in WIRE_COMMAND.items()}

#: `/pubmed` is the one command that needs the caller's own words.
PUBMED_QUERY = "total knee arthroplasty readmission risk factors"

#: The command table's own declarations, keyed by harness key: the primary tool plus any
#: supplementary call. Derived, so "the command did not run" cannot be asserted against a
#: hand-copied list that says something else.
_BY_WIRE = {command.wire: command for command in agent_routing.COMMANDS}
COMMAND_TOOLS: dict[str, set[str]] = {
    key: {_BY_WIRE[wire].tool, *(spec.tool for spec in _BY_WIRE[wire].supplementary)}
    for key, wire in WIRE_COMMAND.items()
    if wire in _BY_WIRE
}

#: The second question of each case, free text on purpose: it refers back to the first
#: answer with a demonstrative, so answering it shows the earlier turn was sent and used.
FOLLOW_UPS: dict[str, str] = {
    "ehr": "What was the discharge disposition and follow-up plan?",
    "meds": "Are any high-risk or anticoagulant medications listed on discharge?",
    "labs": "Which of those laboratory values was most abnormal at discharge?",
    "pubmed": "Which of those findings applies to this admission?",
}


def question_for(prompt: str) -> str:
    """The first question: the literal slash command."""
    if prompt == "pubmed":
        return f"{SLASH_COMMANDS[prompt]} {PUBMED_QUERY}"
    return SLASH_COMMANDS[prompt]


def follow_up_for(prompt: str) -> str:
    """The second question of the case, which is not addressed to an admission."""
    return FOLLOW_UPS[prompt]
