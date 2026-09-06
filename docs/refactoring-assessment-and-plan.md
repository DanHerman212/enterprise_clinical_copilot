# Refactoring Assessment and Plan

**Scope:** `enterprise_clinical_copilot` and the companion `danielmherman` Django repository.

**Purpose:** Establish a maintainable learning path for understanding and improving the existing full-stack AI application without discarding its working behavior, evaluation lineage, or deployment topology.

## Assessment

The application is a functioning distributed ML and agent system. Its complexity is partly inherent:

```text
Django web application
  -> private agent service
    -> LangGraph orchestration
      -> MCP client and server
        -> prediction tool -> Vertex AI endpoint
        -> retrieval tool -> Vector Search and BigQuery
```

The offline path adds the necessary data and model workflows:

```text
MIMIC-IV -> Dataform and BigQuery -> Vertex training pipeline -> model bundle -> endpoint
Discharge notes -> chunking and embeddings -> Vector Search -> retrieval tool
```

A rewrite is not justified by the current evidence. The major runtime boundaries are defensible and the system has already produced evaluation results and a working deployment. Rebuilding would risk losing the relationship between the evaluated artifacts and the running implementation while retaining most of the underlying complexity.

The primary problem is **cognitive and contractual debt**, not a failed architecture. An engineer must currently infer important interfaces from several modules, ordinary dictionaries, environment variables, duplicated parsing logic, and historical documentation.

## What should be preserved

- Django as the public boundary for authentication, synthetic-cohort authorization, quota, and presentation.
- The private agent service as the boundary for LLM execution and response generation.
- MCP as the reusable tool boundary for prediction and retrieval.
- LangGraph as the explicit orchestration implementation.
- Prediction and retrieval as separate domains with separate data paths.
- The MLOps workflow and its evaluation lineage.
- Layered patient isolation and pre-spend authorization.
- The distinction between MIMIC-IV training/evaluation data and the synthetic public demo cohort.
- Optional Langfuse tracing.

These decisions can be reconsidered later, but they should not be removed as a first response to difficulty understanding the code.

## Main sources of debt

### 1. Implicit service contracts

The Django application, agent service, MCP tools, and browser assume response shapes such as `answer`, `tool_calls`, `a2ui`, `citation_map`, `remaining`, prediction fields, and retrieval fields. These contracts are distributed across `demo/views.py`, `demo/agent_client.py`, `services/agent/http.py`, the MCP tools, and JavaScript. They are not represented in one typed contract module.

### 2. Distributed configuration

Project, region, endpoint, table, transport, timeout, and service URL settings are resolved in multiple modules. A configuration error may be discovered on the first request rather than during service startup.

### 3. Presentation and citation logic crosses boundaries

Citation renumbering and mapping occur in Django. Section interpretation is duplicated in Django and browser JavaScript. A2UI composition is server-side while A2UI rendering and additional passage resolution occur in the browser. This is functional, but it is the hardest part of the request path to understand.

### 4. Duplicated domain vocabulary

Section aliases and extraction rules are represented in more than one repository. Feature and serving metadata also have several consumers that must agree on ordering and shape.

### 5. Historical material competes with current implementation

The repository has accumulated design notes, build guides, experiments, generated artifacts, and superseded architecture descriptions. Even when archived, remaining documentation must identify one current source of truth per subsystem.

## Target mental model

The maintainable target is not necessarily a new directory tree. It is four explicit runtime contracts:

| Boundary | Owner | Contract |
|---|---|---|
| Web | Django and browser | Authenticated request, user-facing response, A2UI payload |
| Agent | Django and private agent service | `POST /ask`, timeout, success response, and failure response |
| Tools | Agent and MCP server | Tool schemas, structured results, and structured errors |
| Domains | MCP tools and cloud dependencies | Feature vector, prediction result, retrieved passages, and isolation rules |

The first implementation target is the **agent boundary** because it is the center of the online system. Once that contract is explicit, the downstream prediction and retrieval paths can be understood independently.

## Refactoring principles

1. **Preserve behavior first.** Refactors must not change model outputs, retrieval isolation, authorization order, quota semantics, or public response behavior without a separate decision.
2. **One boundary at a time.** Each change should have one owner, one focused test, and a reversible diff.
3. **Make contracts visible before extracting abstractions.** Do not create a shared package merely to move code; first define what crosses the boundary.
4. **Prefer typed data at service boundaries.** Internal implementation may remain simple, but request, response, tool, prediction, retrieval, and error shapes should be explicit.
5. **Move failures earlier.** Validate configuration and serving metadata at startup or deployment checks rather than on the first user request.
6. **Keep security controls layered.** Django authorization, MCP validation, and retrieval admission checks have different responsibilities; consolidation must not remove defense in depth.
7. **Use source and tests as authority.** Existing prose is a navigation aid, not evidence when it conflicts with code or artifacts.
8. **Do not refactor infrastructure speculatively.** No service merge, protocol removal, model retraining, or deployment change is part of the first phase.

## Staged plan

### Phase 0: Baseline and orientation

**Goal:** Make the current system inspectable before changing it.

- Record the current `POST /ask` behavior, including request fields, success fields, errors, timeouts, and observability fields.
- Identify the current tests for Django, the agent, MCP, prediction, and retrieval.
- Record the current deployment/configuration inputs without printing secrets.
- Mark current source-of-truth files for the feature schema, serving bundle, tool schemas, and response composition.
- Keep historical and generated material out of the primary reading path.

**Exit condition:** An engineer can trace one request from Django to the agent and back using one short map and a known set of source files.

### Phase 1: Make the Django-agent contract explicit

**Goal:** Define the central online interface without changing its wire behavior.

Start with:

- `danielmherman/demo/agent_client.py`
- `enterprise_clinical_copilot/services/agent/http.py`
- `danielmherman/demo/views.py`
- the relevant Django and agent tests

Define and test:

```text
AgentRequest
  question: non-empty string

AgentSuccess
  question
  answer
  tool_calls
  a2ui
  model
  mcp_transport
  optional guardrail and trace fields

AgentFailure
  stable error code
  safe user-facing message
  optional correlation id
```

The first refactor should introduce validation at the boundary while preserving the current JSON keys. It should also make timeout and authentication behavior easy to locate.

**Exit condition:** Focused tests cover valid requests, malformed JSON, empty and oversized questions, unauthenticated service calls, agent timeout, malformed agent responses, and successful response parsing.

### Phase 2: Make tool contracts explicit

**Goal:** Remove dictionary assumptions from the agent and Django integration.

Define schemas for:

- `predict_readmission` result and error.
- `rag_search` result and error.
- `rag_search_sections` result and error.
- recorded tool calls.

Keep MCP as the transport and reuse boundary. The purpose is not to remove MCP; it is to make MCP results understandable without reading every consumer.

**Exit condition:** Tool tests validate success and error payloads, and the agent plus Django use the same contract definitions or validated adapters.

### Phase 3: Centralize runtime configuration

**Goal:** Make deployment assumptions visible and validate them before traffic.

Create one runtime configuration surface for the agent/MCP system. It should own project and region, table references, endpoint and index names, transport, service URLs, timeouts, and tool limits.

Add startup validation for malformed table references, invalid transports, missing required service URLs, invalid timeout values, and malformed serving metadata.

**Exit condition:** A misconfigured local process fails with an actionable startup error, and the same configuration vocabulary is used by local and Cloud Run execution.

### Phase 4: Simplify response composition

**Goal:** Give each response concern one owner.

Recommended ownership:

| Concern | Owner |
|---|---|
| Tool result schema | MCP/domain contract |
| Agent prose | Agent service |
| Citation references | Agent response contract |
| Citation-to-passage resolution | One server-side response adapter |
| A2UI envelope | Django presentation adapter |
| Rendering | Browser/A2UI client |
| Section vocabulary | Retrieval subsystem |

The likely improvement is to have the agent return stable passage identifiers or structured citation references, rather than requiring Django and JavaScript to infer meaning from citation numbers and section aliases.

**Exit condition:** Citation mapping has one canonical implementation and focused tests cover single-section, multi-section, missing-section, and malformed-citation cases.

### Phase 5: Consolidate duplicate domain utilities

**Goal:** Reduce maintenance risk after contracts are stable.

Candidate extractions include:

- shared project and Cloud Run authentication configuration;
- serving bundle discovery and manifest validation;
- feature-source access and vector assembly;
- structured tool errors;
- canonical section aliases.

Do not create a shared module until two consumers have a tested common contract. This prevents a new abstraction from becoming another source of indirection.

**Exit condition:** Duplicate implementations are removed or explicitly justified, and smoke tests continue to exercise the same behavior.

### Phase 6: Documentation and repository cleanup

**Goal:** Make the architecture easy to disseminate to another engineer.

Maintain a small primary documentation set:

- system overview;
- runtime contracts;
- data and model lifecycle;
- retrieval and agent behavior;
- evaluation and limitations;
- operations and data-use constraints.

Archive or remove documents that are historical, superseded, generated, or duplicate. Every retained document should identify its source code and evaluation artifacts.

## Learning sequence for side-by-side work

Each working session should cover one boundary:

1. Django `POST /ask` contract.
2. Agent HTTP service and timeout/error behavior.
3. LangGraph state and tool loop.
4. MCP client transport and schema adaptation.
5. MCP server registration and tool contracts.
6. Prediction tool and model serving bundle.
7. Retrieval tool and patient isolation.
8. Django response composition and A2UI rendering.
9. Offline feature and training lineage.
10. Evaluation and operational controls.

For each session:

1. Read the owning module and its tests.
2. Draw the input/output contract in plain language.
3. Identify one invariant and one failure mode.
4. Add or improve one focused test.
5. Make one small refactor only if the contract is understood.
6. Run the focused validation and review the diff.

## Not part of the initial plan

- Rewriting the application.
- Removing MCP or LangGraph.
- Replacing Django or A2UI.
- Merging the repositories.
- Retraining the model.
- Changing Cloud Run, Vertex AI, Vector Search, IAM, or billing configuration.
- Refactoring based only on aesthetic preference.

Those decisions may become reasonable after the contracts and runtime behavior are understood. They are not prerequisites for reducing cognitive debt.

## First next action

The next working session should begin with a read-only contract map for `POST /ask`. It should identify the exact request and response shapes in both repositories and the smallest existing test set that can protect them. Only after that review should a schema or code refactor be proposed.
