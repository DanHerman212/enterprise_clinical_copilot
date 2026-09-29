# Layer 11 — Security & identity

Status: first document for this layer, 2026-09-29. It had none. Requirement statuses below were
verified against the repository, the settings, and the live project on 2026-09-29.

---

## 1. The layer

Security here is the set of boundaries that remain correct when every participant acts in its own
interest. That framing is not rhetorical: it determines which controls count. A control that
depends on the client behaving well is a convenience; a control that holds when the client is
hostile is a boundary. The layer is therefore organised around trust boundaries rather than
around features, and the question asked of each is what an adversary on the far side can reach.

**Terms.**

| Term | Definition |
|---|---|
| Trust boundary | A line across which data or control passes between parties with different interests, and at which validation must therefore occur. |
| Least privilege | Each identity holds the minimum permissions its function requires, and no other identity can borrow them. |
| Workload identity | An identity attached to a running workload, so credentials are issued to the workload rather than embedded in it. |
| Identity token | A signed assertion of *who* the caller is, scoped to an audience; used for service-to-service authorisation, as distinct from an access token, which asserts what the caller may do. |
| Prompt injection, direct | An instruction supplied by the user, in the channel the user controls. |
| Prompt injection, indirect | An instruction carried in content the system retrieved, arriving through a channel the user does not control and the system treats as evidence. |
| Input screening | Inspection of content before it reaches the model. |
| Output screening | Inspection of the model's answer before it reaches the user. |
| Prompt as code | Instructions held in version control, reviewed like source, and identified by revision, rather than configured in place. |
| Data minimisation | Collecting and retaining the least data the function requires, and synthesising or de-identifying where possible. |
| Supply chain | The path from source to running artifact — build, registry, admission — and the controls that make each step accountable. |
| Human in the loop | A mandatory human decision between the system's output and the consequence it would otherwise carry. |

**The structural asymmetry of this layer.** Layers 1 to 8 describe what the system does; this
layer describes what it may do, and to whom. Two consequences follow. First, security requirements
are rarely satisfied by a component — they are satisfied by an arrangement, which is why least
privilege is assessed across the set of identities rather than one at a time. Second, in a system
that reads untrusted text into a model's context, the *data* path is a trust boundary as much as
the request path: retrieved note text is content an outsider could in principle have authored, and
it reaches the model in the position normally reserved for instructions.

```mermaid
flowchart TB
  U["Anonymous user<br/>UNTRUSTED"] -->|"HTTPS"| EDGE["Edge<br/>LB · Cloud Armor"]
  EDGE --> SITE["Django BFF<br/>website-sa<br/>ONLY PUBLIC SURFACE"]
  SITE -->|"identity token,<br/>audience = service URL"| AG["Agent<br/>agent-sa<br/>IAM-private"]
  AG -->|"identity token"| MCP["MCP server<br/>mcp-server-sa<br/>IAM-private"]
  MCP -->|"IAM + BigQuery IAM"| DATA["Vertex endpoint ·<br/>Vector Search · BigQuery"]
  AG -->|"pinned model"| MODEL["Model gateway"]
  NOTES["Note text<br/>RETRIEVED, UNTRUSTED"] --> MCP
  NOTES -.->|"framed as data,<br/>delimiters escaped"| AG
  BUILD["Cloud Build<br/>cicd-deployer"] -->|"image by digest"| AG
  SEC[("Secret Manager")] -.-> SITE
```

**Boundaries.** Layer 2 owns the front door — the load balancer, Cloud Armor and the
authentication of visitors — and this layer states the requirement it satisfies. Layer 8 owns what
conversation state is kept; this layer owns what may be written to a log at all, which is why the
execution record keeps the shape of a question rather than its text. Layer 9's adversarial cases
are this layer's evidence: they are the only measurement of whether the boundaries hold under
hostile input.

---

## 2. Requirements

| # | Requirement (Google, MUST) | Status |
|---|---|---|
| G1 | Treat all inputs as untrusted — user, retrieved documents, tool results — and filter or validate external content before it enters a prompt. | **Met in part.** Both request boundaries are closed: the BFF refuses any field outside an approved set, and the agent parses a closed contract with a length cap, so an unexpected field is refused rather than silently dropped. Cross-patient isolation is enforced at the source — the `hadm_id` restriction is passed into the index query itself rather than applied to its results, so a retrieval cannot cross patients and then be filtered. Retrieved text is framed as evidence with its delimiters escaped, and tool results are re-validated. What is absent is screening of retrieved content *before* it enters the prompt: the mitigation is framing, not filtering. |
| G2 | Layered defence: screen prompts and responses for injection, jailbreak and harmful content. | **Met in part.** There is no screening service (Model Armor or equivalent). The layers that exist are different in kind: the model's own content filters, whose triggered categories are recorded; a post-hoc guardrail that removes any claim the tool evidence does not support; and twenty adversarial probes judged against explicit criteria. That is three independent defences against a fluent wrong answer, and none of them is a prompt-injection screener. |
| G3 | Prompts treated as code: versioned in Git, centrally stored, reviewed, auditable. | **Met.** The prompt lives in the chain repository, versioned with the orchestration code, the tool wiring and the model pin as one artifact, and the running revision is stamped into every execution record. The question wording was deliberately moved out of the website repository for this reason: half a prompt in a second repository is a prompt that can change without touching the chain, its revision, or its review. |
| G4 | Least-privilege service accounts per service; service-to-service authentication with workload identity, not shared keys. | **Met.** Distinct identities exist per function — the website, the agent, the MCP server, the pipeline, the Langfuse stack, and a dedicated build account, with builds explicitly forbidden from running as the project default. Service-to-service calls carry an identity token whose audience is the callee's service URL, and the agent and MCP services refuse unauthenticated requests at the IAM layer. No shared key is used between services. |
| G5 | Front door: external Application Load Balancer, default `run.app` URL disabled, Cloud Armor for rate limiting and DDoS, IAP or Identity Platform for users. | **Unmet as configured today.** The load balancer and Cloud Armor were implemented and verified live on 2026-09-13, and are torn down between uses under the project's standing decision to minimise billable resources. The site is therefore served directly by its Cloud Run URL, and that default `run.app` endpoint is not disabled — the service's allowed hosts include it. User authentication is Django's own, which satisfies the requirement's intent for a demo but is not the Identity Platform arrangement the requirement names. |
| G6 | Data minimisation: prefer synthetic or de-identified data; do not collect what the product does not need. | **Met.** The public surface serves synthetic data only; MIMIC-IV feeds training and evaluation and never ships. Question and answer text is excluded from logs by construction; conversation state is retained for twenty-four hours and swept; secrets are held in Secret Manager with the previously tracked credential file removed from version control. Two checklist items remain open and are recorded under the gaps below. |
| G7 | Supply chain: images built by Cloud Build, scanned in Artifact Registry, only approved images deployable. | **Met in part.** Images are built by Cloud Build under a dedicated least-privilege account and pushed to Artifact Registry, tagged by commit digest so the running artifact is identifiable. There is no vulnerability scanning configuration and no Binary Authorization policy, so *which* image may deploy is decided by whoever holds deploy permission rather than by admission control. |
| G8 | Red-team against the OWASP LLM Top 10; regular audits. | **Met in part.** A nine-part adversarial code review of the source, and twenty adversarial behavioural probes against the deployed service across eight criteria families, with one failure recorded. Neither is mapped to the OWASP LLM Top 10 taxonomy, and injection is exercised through the question channel only — the indirect variant, an instruction embedded in retrieved note text, is recorded as uncovered because it requires a corpus change. |
| G9 | AI-aware incident response playbook: roll back the model or prompt version, disable the endpoint, isolate the data source. | **Unmet.** The mechanisms exist as scripts — a teardown for the serving endpoints, a deploy path that verifies before promoting, and Blue/Green promotion for the index — but there is no playbook that says which to use, in what order, for which class of incident, and no rehearsal of it. |

---

## 3. Gaps and recommended remediation

**G1 — no screening of retrieved content before it enters the prompt (G1, G2).** Note text arrives
as evidence, in the position where instructions normally appear. *Remediation:* adopt a screening
service at the retrieval boundary, or implement the equivalent as a validated filter over retrieved
passages. The stronger half of this is already built and should not be given away: the delimiter
escape prevents a passage from *closing* its own quotation, and the adversarial criterion
`must_not_leak_other_patient` is scoped to exactly this risk. The gap is that a passage can still
carry a complete, well-formed instruction and nothing inspects it.

**G2 — no admission control on images (G7).** *Remediation:* enable vulnerability scanning on the
registry, and add a Binary Authorization policy requiring attestation before deploy. The build
already runs under a dedicated account with a narrow role set, so the policy has a subject to
attest.

**G3 — no incident response playbook (G9).** *Remediation:* a short document with three named
scenarios — a bad answer attributable to a prompt or model change, a retrieval incident, and a
suspected data-exposure event — each mapping to existing scripts, with the rollback target
identified and the decision owner named. The site's build already deploys a revision with no
traffic and promotes only after migrations succeed, which makes rollback "do not promote, or
promote the previous revision"; that property should be written down rather than rediscovered.

**G4 — red-team coverage is unclassified and incomplete (G8).** *Remediation:* map the existing
twenty probes and the nine code-review sections to the OWASP LLM Top 10 categories, so the
coverage that exists becomes legible as coverage and the holes become legible as holes. Then close
the indirect-injection case, which is the same corpus change the evaluation layer records as its
own gap 4 — one action satisfies both layers.

**G5 — the front door is torn down (G5).** *Remediation:* a decision rather than a task, and it
should be taken explicitly: either the load balancer and Cloud Armor are stood up for any window
in which the demonstration is publicly reachable, or the cloud-run default hostname is accepted as
the front door on the record. What is not defensible is the current state, in which the control is
implemented, documented as verified, and absent at the moment a visitor arrives.

**G6 — two open items on the compliance checklist (G6).** Evaluation report files containing
admission identifiers and clinical fragments are not yet scrubbed or gitignored, and PhysioNet and
MIMIC-IV are not yet acknowledged in any published write-up. *Remediation:* both are small and
neither is technical; the first is a repository-hygiene change, the second is owed on publication.

---

## 4. Record of change

Entries are added as gaps close.

No change has been made to the system for this layer. This document records its state on
2026-09-29; the requirement statuses above are the baseline against which the entries that follow
will be measured.
