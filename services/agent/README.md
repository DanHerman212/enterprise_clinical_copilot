# `services/agent` — the clinical copilot's reasoning service

The diagram is the request path, read left to right. The main line is the sequence a question
passes through; the boxes grouped beneath it are the modules each step calls. The **record**
sits off the path on purpose — one line is written for every run, including the ones that
never reach an answer.

The package is six subpackages in a single direction: `ingress` → `turn` → `engine` →
`postprocess` → `presentation`, with `routing` and `telemetry` called from anywhere and
calling back into nothing above them. `tests/agent/test_layer_separation.py` holds that
import DAG acyclic; the arrows below are the DAG, not a drawing convention.

```mermaid
flowchart LR
  caller["Django site"] --> http

  subgraph ingress["ingress — the surface"]
    http["http.py"]
    contracts["contracts.py"]
  end

  subgraph turnpkg["turn — one turn, transport-blind"]
    turn["turn.py"]
  end

  subgraph routing["routing — the question"]
    route["routing.py"]
    admctx["admission_context.py"]
  end

  subgraph engine["engine — the run"]
    graph["graph.py"]
    deterministic["deterministic.py"]
    prompts["prompts.py"]
    model["model.py"]
    mcp["mcp_client.py"]
  end

  subgraph postprocess["postprocess — what may be served"]
    guard["guardrail.py"]
    cites["citations.py"]
    follow["followups.py"]
  end

  subgraph presentation["presentation — what it looks like"]
    a2ui["a2ui.py"]
    canvas["canvas.py"]
    domain["domain.py"]
  end

  subgraph telemetry["telemetry — the record"]
    obs["observability.py"]
    stages["stages.py"]
  end

  http --> turn --> graph --> guard --> a2ui --> out["the answer"]
  contracts -.-> http
  route -.-> graph
  admctx -.-> turn
  prompts -.-> graph
  deterministic -.-> graph
  model -.-> graph
  mcp <--> graph
  mcp <--> tools["MCP server"]
  cites -.-> guard
  cites -.-> a2ui
  canvas -.-> a2ui
  domain -.-> canvas
  follow -.-> a2ui
  stages -.-> turn
  obs -.-> turn
```

## Modules

| Path | What it is for |
|---|---|
| **`turn.py`** | One turn, from a parsed request to a validated payload: the deadline, the tool session, pre-flight hydration, the guardrails, the presentation, the execution record. Both routes call in here, so `/ask` and `/ask/stream` cannot differ in what they accept or when the checks run. |
| **`ingress/`** | Everything that knows about a transport. |
| `ingress/http.py` | The service surface: two routes, one answer and one stream, plus a health check. Owns IAM, status codes, SSE framing and keepalives, and nothing about the answer. |
| `ingress/contracts.py` | The request and response contracts. Refuses anything outside the agreed shape, before a model runs and after. |
| `ingress/run.py` | The developer entry point for asking a question by hand. Not part of the request path. |
| **`engine/`** | The run itself: how a turn is asked for, what it may call, and how it ends. |
| `engine/graph.py` | The graph: the model turns to choose a tool, the tool runs, the model turns again to narrate the result. |
| `engine/deterministic.py` | The command path. A slash command names its tool; the model is not asked and has no tools bound, so a secondary search is impossible rather than forbidden. |
| `engine/mcp_client.py` | The bridge to the tool server, presenting its tools to the model as ordinary tools. Holds the call budget, the once-per-turn set and the result rendering. |
| `engine/prompts.py` | The system prompt, and with it the rules the model runs under. |
| `engine/model.py` | How a turn ended — tokens, serving model, safety filters, and the reason an answer was cut short. |
| **`routing/`** | What the user pressed or typed, turned into what the model is asked. |
| `routing/routing.py` | The wording of the question: a chip or a typed sentence becomes the text the model receives, and a command becomes its tool and arguments. |
| `routing/admission_context.py` | The admission's structured features, fetched once by the caller and stated to the model as verified ground truth, so a fact already in hand costs no tool call and needs no query. |
| **`postprocess/`** | The deterministic checks after the model writes. The model proposes; this decides what may be served. |
| `postprocess/guardrail.py` | The gate. An unsupported number is removed, an invented age redacted, a medication absent from the retrieved notes stripped. |
| `postprocess/citations.py` | Citation reconciliation: the model's markers are renumbered against the passages actually retrieved and resolved to note sections. |
| `postprocess/followups.py` | The follow-up chips, computed from the chart facts the turn already holds. Zero network calls: a chip is a control, so what it asks is a property of the record rather than of the model's imagination. |
| **`presentation/`** | What the turn's evidence is drawn as, and the envelope it travels in. |
| `presentation/a2ui.py` | The composer. Which calls were this turn's evidence, and what goes on the wire. |
| `presentation/canvas.py` | The drawing: the estimate's band, the cohort table, the medication grid, the lab trajectories, and the ordering and unavailable-states that decide which a given turn gets. |
| `presentation/domain.py` | The clinical vocabulary — feature labels, note sections, article fields — so the chart and the prose beside it cannot disagree. |
| `presentation/__init__.py` | The envelope constants. They live here because the composer imports the canvas, so placing them in either would make the pair circular. |
| **`telemetry/`** | The record. Off the request path; the service runs unchanged when tracing is switched off. |
| `telemetry/stages.py` | Progress: what the run is doing, turned into the events a waiting caller sees. |
| `telemetry/observability.py` | Tracing as a sink, never a dependency — the chain's identity, and one span per execution. |
