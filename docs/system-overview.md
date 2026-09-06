# System Overview

Enterprise Clinical Copilot is a cloud-backed clinical decision-support application. It
combines a structured readmission-risk model with retrieval-grounded agent responses.
The numerical estimate and the narrative evidence are deliberately separate: the model
scores structured features, while retrieved discharge-note passages support the agent's
explanation.

```text
Django application
  -> services/agent       HTTP boundary, LangGraph, Gemini
    -> services/mcp       MCP server and domain tools
      -> Vertex endpoint  prediction and TreeSHAP
      -> BigQuery         features and note text
      -> Vector Search    semantic note retrieval

Dataform -> BigQuery -> mlops/training -> mlops/serving
Notes   -> services/mcp/pipelines -> Vector Search
```

## Repository boundaries

| Boundary | Location | Responsibility |
|---|---|---|
| Public application | Companion Django repository | Login, demo authorization, quota, and A2UI presentation |
| Agent service | `services/agent/` | `/ask`, LangGraph execution, prompts, guardrails, observability |
| MCP service | `services/mcp/` | Tool registration, prediction, retrieval, and cloud dependencies |
| Model lifecycle | `mlops/` | Feature schema, training, evaluation, serving, and deployment |
| Data transformation | `definitions/` | Dataform cohort, labels, splits, marts, and assertions |
| Evaluation | `evaluation/` | Agent traces, retrieval measurements, judging, and reports |

## Request path

Django validates the authenticated request and the synthetic admission allowlist before
claiming quota. It sends one question to the private agent service. The agent invokes MCP
tools as needed, receives structured results or structured errors, and generates the final
answer. Django adds presentation-specific citation and A2UI data before returning the
response to the browser.

The public demo uses a synthetic cohort. MIMIC-IV is used for training and evaluation under
its data use agreement. The system is not a validated medical device and is not intended
for clinical use.
