# `services/agent` — the clinical copilot's reasoning service

One question in, one answer out. It receives a question from the website, runs a model
against the patient's own notes and prediction tools, checks the answer deterministically,
and returns the answer with the evidence it rests on.

The diagram is the request path, read top to bottom: **request** → **run** → **answer**, with
the **record** written alongside it. The record sits off the main path on purpose — one line
is written for every run, including the ones that never reach an answer.

```mermaid
flowchart TD
  caller["Django site"]

  subgraph req["Request"]
    http["http.py"] --> contracts["contracts.py"] --> questions["questions.py"]
  end

  subgraph run["Run"]
    prompts["prompts.py"] --> flow["graph.py"]
    flow <--> mcp["mcp_client.py"]
    flow --> stages["stages.py"]
  end

  subgraph ans["Answer"]
    turn["model_turn.py"] --> guard["guardrail.py"] --> cites["citations.py"] --> canvas["a2ui.py"]
    labels["feature_labels.py"] --> canvas
  end

  subgraph rec["Record"]
    chain["chain.py"]
    obs["observability.py"]
  end

  tools["MCP server"]

  caller --> http
  questions --> flow
  flow --> turn
  mcp <--> tools
  canvas --> chain
  flow -.-> obs
```
```

## Modules

| Module | What it is for |
|---|---|
| `http.py` | The service surface: two routes, one answer and one stream, plus a health check. Owns the deadline and the per-request tool session. |
| `contracts.py` | The request and response contracts. Refuses anything outside the agreed shape, before a model runs and after. |
| `questions.py` | The wording of the question: a chip or a typed sentence becomes the text the model is asked. |
| `graph.py` | The run itself: the model turns to choose a tool, the tool runs, the model turns again to narrate the result. |
| `prompts.py` | The system prompt, and with it the rules the model runs under. |
| `mcp_client.py` | The bridge to the tool server, presenting its tools to the model as ordinary tools. |
| `stages.py` | Progress: what the run is doing, turned into the events a waiting caller sees. |
| `model_turn.py` | How a turn ended — tokens, serving model, safety filters, and the reason an answer was cut short. |
| `guardrail.py` | The deterministic checks after the model writes. The model proposes; this decides what may be served. |
| `citations.py` | Citation reconciliation: the model's markers are renumbered against the passages actually retrieved and resolved to note sections. |
| `a2ui.py` | The presentation adapter. Composes the canvas, the source list and the literature from the guarded answer and the tool evidence. |
| `feature_labels.py` | The one place a model feature is named, so the chart and the prose beside it cannot disagree. |
| `chain.py` | The chain's identity and its record: which model, which revision, one line per execution. |
| `observability.py` | Tracing as a sink, never a dependency. The service runs unchanged when it is switched off. |
| `run.py` | The developer entry point for asking a question by hand. Not part of the request path. |
