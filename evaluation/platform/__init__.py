"""Platform-owned evaluation (Langfuse datasets + experiments).

Step order matters and is enforced by tests, not by good intentions:

    schemas    -> the declared contract every other module validates against
    fixtures   -> datasets seeded from files in the repo
    task       -> one agent round-trip per item (two turns, one session)
    evaluators -> assertions, then the rubric judge, then the pass rule, then the gate
"""

from evaluation.platform.schemas import DatasetItemInput, TaskOutput

__all__ = ["DatasetItemInput", "TaskOutput"]