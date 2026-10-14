# 02B — Streaming: the rethink

Status: position paper. No code has been written for this. Tabled to Phase 4 by the
product owner on 2026-10-14.

---

## 1. Why this document exists

The approved Phase 2 specification required that the narrative prose — from a slash
command and from free text alike — stream to the canvas token-by-token, with the
structured visual components appearing as their payloads complete. Time to first token
was named as the reason.

What was delivered instead is progress-only: the clinician watches which step is
running, and the answer arrives complete in a single frame. At no point in the
configuration that reached production did any prose stream.

Two failures, and they are worth separating because only the second is a delivery
problem.

The **design** failure: a per-path streaming policy was written into the implementation
in which a single command qualified to stream, and the rest were buffered. The reasoning
behind it is sound and is set out in section 3 — a safety pass rewrites the text after
the model finishes. But a narrowing of an approved requirement was decided at
implementation level and never raised as a deviation, so the requirement was recorded as
met while being unmet.

The **delivery** failure: when the gap was reported, work began before the shape of the
fix was agreed. The build was stopped before anything was written, but the sequence was
wrong. The guard against both is the same: the design decision goes to the product owner
before code, and delivery is checked against the requirement list rather than against
the policy that quietly replaced it.

---

## 2. The requirement as it now stands

Streaming is a product requirement, not an optimisation. All paths. The chosen shape is
to stream provisional text and swap in the guarded text, with the resolved citations,
when the payload completes; where the safety pass alters or removes something, the
correction is surfaced through the evidence-check affordance so the reader is not left
wondering whether they misread.

That last clause is the interesting one, and it is what makes the design tractable: the
product owner has accepted that a correction may follow text on screen, provided the
correction is declared. The question is therefore no longer "how do we stream without
ever showing a corrected draft" — it is "how do we stream, and how do we declare a
correction clearly".

---

## 3. The constraints any design must satisfy

These were established by reading the implementation, and each one rules something out.

**Three sources of post-hoc mutation, not one.** The guardrails rewrite the text after
the model finishes — removing a dose or frequency the retrieved note does not support,
removing a risk number that does not match the prediction, and removing a citation
marker that resolves to nothing. Alongside them, a reply that arrives wrapped in the
contract's envelope is unwrapped, so the served prose is not the text the model
produced; and a question the notes do not answer is replaced by a fixed sentence. Any
claim that the streamed text equals the served text is false for all three reasons, not
just the first.

**The medication command cannot stream the model's writing.** Its served prose is
generated in code from the verified rows; the model's formatting text is discarded
before anything downstream sees it. Only the generated narrative can be streamed, and it
does not exist until the tool has returned and the rows have been verified. Streaming
that command means streaming a text that is already final, which is worth doing — it is
the earliest possible start — but it is a different mechanism from streaming tokens.

**Free text cannot know in advance whether a turn is the answer.** The loop hands the
model the question and its evidence; the model either answers or asks for another tool,
and only the finished turn reveals which. To stream between first token and completion
is therefore to accept that a turn which turns out to be a tool decision may have had
text attached to it. In practice a tool-deciding turn carries little or no prose, but the
design must state what happens if it does rather than assume it will not.

**Text on screen must not contradict the served text without saying so.** This is the
rule the rest of the system already follows — an unavailable section says it is
unavailable, a failed search says it failed, a missing value says it is not stated.
Silent substitution is the only behaviour the existing design never permits.

**The plumbing exists.** There is already a frame type for a prose fragment, a stream
route that carries it, a client loop that consumes frames, and a comparison that logs
when what was announced differs from what is served. This is a policy decision and a
swap protocol, not a greenfield build.

---

## 4. Options

**A — progress only.** The status quo. Cheapest, no safety exposure, and no streaming.
Explicitly refused by the product owner, so it is recorded only as the baseline it
departs from.

**B — stream, then swap silently.** The smallest change that satisfies the letter of the
requirement: provisional text on screen, the guarded text replacing it on completion.
The risk is the failure mode that motivated buffering in the first place — a clinician
reads a dose, and the dose disappears with no indication that anything happened. It
treats the swap as a rendering detail when it is an event the reader should be told
about.

**C — stream, swap, and declare the correction.** Recommended. The draft streams; the
terminal frame remains authoritative and replaces it; where the safety pass changed the
text, the evidence-check affordance states which kind of correction was applied. The
affordance already exists, already carries the meaning "this answer was checked against
its evidence", and is the natural place for the one additional fact.

The correction notice can be specific rather than vague, because the pass already
classifies what it did: a dose or frequency not supported by the record, a risk number
that does not match the prediction, a citation that could not be resolved. That is four
or five kinds, so the notice can name one instead of saying "something was adjusted" —
which would teach a reader to distrust the whole answer rather than the part that
changed.

**D — stream only what nothing can correct afterwards.** Measured against the three
mutation sources, this amounts to almost nothing: the literature path resolves its
citation markers after the fact, and the note path is guarded throughout. It is a
defensible answer to the safety objection and a poor answer to the product requirement.

---

## 5. What I would measure before designing

The safety argument for buffering was mine, and it was never quantified. Three
measurements turn it from an argument into a number, and all three are cheap.

1. **How often the announced and served texts differ, per path.** A comparison already
   runs and logs a warning when they diverge. If it fires rarely on the literature and
   note paths, the correction notice will be an occasional event rather than a
   recurring one, and the design can be judged on its normal case.
2. **Time to first token, per path, as it stands.** Streaming has to be shown to
   improve something. Without a baseline, "it feels faster" is the only claim available.
3. **How often a free-text turn writes prose before deciding to call a tool.** This
   decides whether the free-text path ever needs to withdraw a paragraph it has already
   shown, which is the one case where the swap is visible rather than invisible.

---

## 6. If the decision is C, the shape of the build

One change across three layers, in this order, because each layer is verifiable alone.

1. **The agent** announces prose fragments as the model writes them, on every path that
   has prose, whenever the caller asks. The terminal frame does not change shape and
   stays authoritative, so a caller that ignores the fragments still receives exactly
   what it receives today.
2. **The site** asks for those fragments on the console's behalf and relays them. It
   already relays the progress frames and already opens the event stream; this is one
   field in the request and no new machinery.
3. **The client** writes the fragments into the answer element as provisional, marks
   them as provisional, and replaces them with the served answer and its resolved
   citation markup when the terminal frame arrives. If the terminal frame reports that
   the safety pass changed the text, the evidence-check affordance says which kind of
   change was applied.

The medication command takes the second mechanism from section 3: its generated
narrative is announced in fragments as soon as it exists, which is not token streaming
but is the same user-visible outcome one step earlier in the sequence.

---

## 7. Questions for the product owner

1. **Granularity of the notice.** One notice per turn, naming the kinds of correction
   applied, or one notice per correction? I would do the former: the reader wants to
   know the answer was adjusted, and the kinds are the useful detail.
2. **Is the withdrawn text shown?** I would not show it. Showing what was removed
   invites a reader to weigh a claim the system has already rejected.
3. **Does the medication command stream its generated narrative?** I would. It is the
   settled answer and it exists earlier than the envelope that carries it.
4. **Is a free-text turn's pre-tool reasoning shown?** I would suppress it. It is not
   the answer, and it is the one place where a fragment could be retracted mid-screen.
5. **Does the provisional text carry any marking while it is provisional?** A subtle one
   is honest and costs nothing; a prominent one invites the reader to wait for the
   settled version, which defeats the point of streaming it.
