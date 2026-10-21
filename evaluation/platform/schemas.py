"""Declared contracts for platform-owned evaluation.

The legacy harness passed an untyped dictionary through seven hops and four runs each
failed at a different one. These models are that contract, enforced at the two places a
mistake used to cost a full run:

* an item cannot exist without its decoded patient manifest, and
* a task cannot return a blank answer — a slash-command card is serialised to text
  rather than leaving the Langfuse `Output` column empty.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

WorkflowType = Literal["command_plus_followup", "single_probe"]

#: A response that is present but says nothing. Treated as blank, because a card
#: rendered as an em dash is what made the dashboard unreadable.
BLANK_RESPONSES = frozenset({"", "—", "-", "n/a", "none", "null"})


class DeterministicExpectations(BaseModel):
    """What the dataset's `expected_output` holds: envelope facts, all optional.

    Empty for a census item (there is no correct answer to assert), populated for a
    targeted probe that must route to a command, render a component, or refuse with a
    specific status. `structural_check` reads exactly these fields.
    """

    command: str | None = None
    routed_to: str | None = None
    component: str | None = None
    status: int | None = None
    error: str | None = None
    literature_nonempty: bool | None = None
    max_tool_calls: int | None = None
    tool_calls: int | None = None

    #: Fixture expectations carry a few extra descriptive keys; they are kept in the
    #: item input for the record and ignored here rather than rejected.
    model_config = ConfigDict(extra="ignore")


class DatasetItemInput(BaseModel):
    """One dataset item: a patient, a workflow, and the prompts to run.

    `patient_manifest` is mandatory and must be populated at seeding time. It is the
    decoded profile the service gives the agent as pre-flight context, and it is never
    a tool call — so if it is not captured here it does not reach the judge at all.
    """

    patient_id: str
    workflow_type: WorkflowType
    turns: list[str]
    patient_manifest: dict[str, Any]
    #: The command under test (`risk`, `meds`, …) where the item is a command turn.
    command: str | None = None
    #: Deterministic expectations from the fixture (status / component / error …).
    expectations: dict[str, Any] = Field(default_factory=dict)

    @field_validator("patient_id")
    @classmethod
    def _patient_id_present(cls, value: str) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("patient_id is required")
        return text

    @field_validator("patient_manifest")
    @classmethod
    def _manifest_mandatory(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(value, dict) or not value:
            raise ValueError(
                "patient_manifest is mandatory and must be non-empty: it is the only "
                "route by which the pre-flight manifest reaches the judge"
            )
        return value

    @model_validator(mode="after")
    def _turn_count_matches_workflow(self) -> DatasetItemInput:
        if not self.turns:
            raise ValueError("turns must contain at least one prompt")
        expected = 2 if self.workflow_type == "command_plus_followup" else 1
        if len(self.turns) != expected:
            raise ValueError(
                f"{self.workflow_type} needs {expected} turn(s), got {len(self.turns)}"
            )
        return self


class TaskOutput(BaseModel):
    """What `task(item)` returns, and what the evaluators read. Not a dataset field.

    It is enforced in code by `task()`; it is deliberately NOT the dataset's
    `expected_output_schema`. Declaring it there was the first attempt and it failed on
    the server (2026-10-20): a dataset enforcing `expected_output` as a finished answer
    rejects a census item, because no expected answer exists — only expectations about
    the envelope. That is what `DeterministicExpectations` is for.
    """

    turn_1_response: str
    turn_2_response: str | None = None
    tool_evidence: list[dict[str, Any]] = Field(default_factory=list)

    #: Tracing / diagnosis. Additive to the spec: the evaluators need the envelope
    #: facts (status, command, components) to run deterministic assertions without a
    #: second network call.
    session_id: str = ""
    #: The trace the service opened while answering. The experiment writes this item's
    #: scores against it, so a score and its run tag live on the same trace — without
    #: it every score landed on the harness's own untagged wrapper trace (2026-10-21).
    langfuse_trace_id: str = ""
    turn_1_status: int | None = None
    turn_1_command: str | None = None
    turn_1_components: list[str] = Field(default_factory=list)
    turn_1_card: dict[str, Any] | None = None
    turn_1_error: str | None = None
    turn_2_status: int | None = None
    turn_2_trace_id: str = ""

    @field_validator("turn_1_response")
    @classmethod
    def _response_not_blank(cls, value: str) -> str:
        if value.strip().lower() in BLANK_RESPONSES:
            raise ValueError(
                "turn_1_response is blank: serialise the command card to text/JSON so "
                "the trace output is readable"
            )
        return value
