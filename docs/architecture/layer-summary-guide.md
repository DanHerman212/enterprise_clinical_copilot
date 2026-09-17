# Layer summary guide

How to write `layer-NN-summary.md`: one page per layer, for building a mental model of the
architecture. The layer document (`layer-NN-*.md`) stays the source of truth for evidence,
decisions, citations and open questions. This is a summary **on top of** it, for a reader who
wants to understand the layer, not audit it.

## Rules

1. **Sequence:** layer explanation → how it works today → gaps.
2. **Exclude everything that is not this layer.** No "this is covered by layer N", no
   deferrals, no forward references. If a fact does not describe this layer, cut it.
3. **No line-number citations.** The source document carries those.
4. **No decisions log, no interview Q&A, no evidence blocks, no "what we measured".**
   A number belongs here only if it is load-bearing for understanding the layer.
5. **Brevity is the requirement.** Target under ~90 lines including the diagram. Short
   sentences. One idea per line. If a sentence is not load-bearing, delete it.
6. **Technically accurate.** Never state a mechanism the code does not have. If unsure of a
   detail, check the source document or the code before writing it.
7. Write in the present tense for how it works, past tense for gaps.

## 1. The layer

Half a page.

- What it is and why it exists, in the application's terms — what breaks without it.
- A **Mermaid** diagram: components, the flow through them, and the boundary to the layers
  either side.
- Components: a table, one line each, naming the file or service.
- Upstream / downstream: one line each. What calls this layer, what it reads or hands on.
- Terms: 4–6 industry terms actually used below, defined in one line each.

## 2. How it works today

A third of a page. Present-tense prose describing the normal path through the layer, from
the request arriving to the result leaving, plus the behaviour at its edges (what happens on
no data, a bad argument, a dependency failure). No file paths unless a component's identity
depends on it.

## 3. Gaps

One short block per gap, in the order the source document numbers them. Each block:

- **What it was** — one or two sentences, stated as a consequence for the application, not
  as a code defect.
- **The fix** — one or two sentences.
- **Changed** — one line naming the files and tests touched, in plain language.

Close with a compact requirements table: one row per requirement the layer is audited
against, one clause each, stating whether it is met.

## Working with the user

The first draft gets reviewed and the guide gets revised from that feedback. Keep the guide
and the drafts in step: when the user asks for a different shape, edit this file, then the
summaries that follow it.
