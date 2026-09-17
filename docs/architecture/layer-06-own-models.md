# Layer 6 — Your own models

Status: audited 2026-09-17, and independently reviewed the same day. Twelve gaps, all
recorded in section 5 with the decision that closes each; gaps 1 to 10 and 12 are closed,
and gap
11 is recorded with its options and awaits a decision. A
gap is described once — defect, decision, outcome — rather than restated per section.

---

## 1. What the layer is

One model in this system was fitted by us: the 30-day readmission risk model. The
foundation model and the embedding model are Google's; the notes, features and
vectors are data. This layer is the life of that one artifact — the pipeline that
fits it, the gates that decide whether it may be published, the bundle it is
published as, the registry entry that names it, and the endpoint that answers with
it.

It is not the chain that decides to ask for a score, and it is not the tool
contract that wraps the call — those are layers 3 and 5. It is not the feature row
either: the model reads a row from BigQuery, and the rows are layer 7's.

```
  ┌───────────────────────────────────────────────────────────┐
  │  LAYER 5 — the tool boundary                              │   owns the call, the
  │  predict_readmission, tools/prediction.py 118             │   contract, and what
  │  the endpoint handle, dependencies/model_endpoint.py 16   │   the model is asked
  └──────────────────────┬────────────────────────────────────┘
     IN │ one 49-value vector, in an order read from the newest bundle
        v
  ┌───────────────────────────────────────────────────────────┐
  │  LAYER 6 — the model's own lifecycle                      │   owns the fit, the
  │  OFFLINE  Vertex Pipelines, training_pipeline.py 97       │   gates, the bundle,
  │  PUBLISH  Model Registry: register_model.py 294          │   the registry entry
  │  SERVE    the CPR, serving/cpr/predictor.py 68            │   and the deployment
  └──────────────────────┬────────────────────────────────────┘
     IN │ a table rebuilt in place · an image tag
        v
  ┌───────────────────────────────────────────────────────────┐
  │  THE GROUND — not operated here                           │   owns the rows, the
  │  BigQuery analytics_dataset_encoded · GCS bundle dirs ·   │   files and the
  │  Artifact Registry                                        │   images it reads
  └───────────────────────────────────────────────────────────┘
     BACK │ probability, threshold decision and TreeSHAP factors — every number the
          │ chain will state about risk comes from here, together with the
          │ identity of the model that produced it, reported by the endpoint
```

What crosses each boundary, and what does not:

| Boundary | Crosses | Does not cross | Who decides |
|---|---|---|---|
| Pipeline → registry | A bundle, a registry entry naming that bundle, and the gate numbers | A container. Registration publishes artifacts; the servable model is uploaded by a separate script | The pipeline decides that the model may be registered; the deploy script decides what serves |
| Registry → endpoint | The bundle the deploy script discovered, deployed at 0% traffic first, then the traffic shift | The pipeline's own record — a deployment uploads its own model entry, so what serves and what the pipeline registered are two different things | The deploy script, at the moment it runs |
| Endpoint → tool | Probability, threshold decision, TreeSHAP factors in declared units, and the identity of the model that produced them | An answer without an identity: the deployment stamps the model's name into the container and the endpoint reports it with every prediction | The CPR builds the response and names itself; the caller reports what it is told |
| Tool → endpoint | One instance keyed by feature name, in the bundle's feature order | A positional vector — the tool no longer sends one, because a name the endpoint does not know is refused while a reordered vector would simply be scored | The manifest decides which names are sent; the endpoint decides whether it knows them |
| Ground → pipeline | The encoded rows, read once per split through one query | A write: nothing in the pipeline modifies the view it reads | Dataform owns what the view holds; the pipeline owns the read |

Terms used in this document:

| Term | Meaning |
|---|---|
| Pipeline | The Vertex AI (KFP) DAG that fits the model. Twelve steps, one artifact: the compiled or submitted definition. |
| Component | One step of that DAG — a containerised function with typed inputs and outputs. |
| Gate | A step that fails the run rather than returning a number, so an unfit model cannot be registered. |
| Serving bundle | The directory registration publishes: the booster, the feature manifest, the threshold, the gate metrics and the digests. |
| CPR | Custom Prediction Routine — our own prediction container, which loads the bundle and returns a probability, a decision and attributions. |
| Registry entry | A Vertex AI Model resource: a named reference to a bundle and an image. The pipeline writes one as a provenance record; a deployment may serve that record directly. |
| Lineage | The record of what produced an artifact — for a model, which run, which code and which data. |

**Layer 6 is: the life of the one model we trained — how it is fitted, what it must
prove before it is published, and which copy of it answers.**

---

## 2. What Google requires

| # | Requirement | Level |
|---|---|---|
| E5 | Own models (e.g. a classifier) go through a pipeline: train → evaluate → Model Registry → endpoint, with lineage in ML Metadata. | MUST |
| E6 | Retraining / re-tuning triggered by monitoring, not by hand. | SHOULD |
| E1 | Version control for prompt templates, chain code, external datasets, adapter weights and container images. Only the model's own half is audited here: the weights, the dataset, the images. | MUST |

---

## 3. How this application implements it

**The pipeline.** `mlops/training/training_pipeline.py` defines the DAG (107) and is
the deployable artifact: `compile_pipeline` (275) turns it into KFP IR and `submit`
(293) posts that IR to Vertex AI Pipelines with caching off (328), so every run is a
real run rather than a reuse of a cached step. Twelve steps run in order — an
importer that turns the training table into a lineage artifact, then load,
validate, benchmark, gate, HPO, train, calibrate, evaluate, explain, audit,
register — and the feature contract is code, not data: the feature order comes from
`mlops/data/encoding.py` (169), and `load_data` emits the serving manifest from the
same module (110, 164), so the model's own inputs cannot be rearranged by a
configuration change. Categorical encoding is materialised in BigQuery
(`mlops/data/config.py` 66), so the model is a plain numeric booster.

`load_data` also hard-fails when any patient's admissions straddle two splits
(`load_data.py` 47, 165). This matters more than it looks: every gate below it — the
test score, the stability check, the fairness audit — assumes patient-level
disjointness, and an upstream regression in the split column would otherwise leave
all of them
reporting PASS on contaminated evidence.

**The gates.** Four things must hold before a model can be registered, and none of the
four thresholds is an input: they are code, or the versioned artifact the image
carries, so no submitter can set one.

- *Drift.* Evidently compares train against validation and fails the run when the
  drifted-column share exceeds 20% (`validate_data.py` 60). The 20% is a constant
  with its reasoning written beside it (`_baselines.py` 28–33), not a parameter.
- *Benchmark.* A default-parameter XGBoost must beat the HOSPITAL baseline AUCPR by
  at least 0.01 (`benchmark_gate.py` 20, 41). The baseline is read inside the gate
  from the artifact shipped in the training image (`_baselines.py` 36–57), and a
  missing or implausible one stops the gate (`benchmark_gate.py` 23) — which is what
  the earlier version of this gate could not do while the value arrived as
  `hospital_aucpr=0.0`.
- *Test.* The held-out split must beat HOSPITAL by the same margin and must not
  degrade by more than 0.02 AUCPR from the HPO validation estimate. Both failures
  raise rather than print (`evaluate_test.py` 39, 105, 114), and this gate reads the
  same artifact as the benchmark gate, so the two cannot be handed two different
  numbers.
- *Order.* Registration is wired to run after the eval gate and both audits
  (`training_pipeline.py` 257), so a crash in an audit cannot leave an
  already-registered model with no audit artifacts.

The fairness audit is deliberately not a gate. It reports TPR and FPR parity per
subgroup with Wilson intervals, marks itself advisory, and hands the decision to a
human (`fairness_audit.py` 29) — the stated reason is that small subgroups and
drift would otherwise break builds on noise.

**The bundle.** Registration publishes a directory rather than a container:
`model.bst` (the native booster), `manifest.json` (feature order, the
one-hot → parent-group map used to aggregate attributions, and the reference to the
data it was fitted on), `threshold.json` (the
operating threshold), `gate_metrics.json` (the gate numbers) and `checksums.json`
(SHA-256 per file) — `register_model.py` 44–107. The booster is written in
XGBoost's own format, and the components that hand a *fitted model* between steps
verify a SHA-256 sidecar before deserialising it (`_artifact_integrity.py` 20, 29),
because that path is pickle and pickle is code execution. The digest is written by
the same process and into the same directory as the artifact, so it detects
corruption rather than tampering.

The endpoint refuses to serve a bundle it cannot verify: a missing `checksums.json`,
a missing listed file, or a digest mismatch all fail startup
(`predictor.py` 40–64). It also refuses to start without `threshold.json` rather
than falling back to 0.5 (`predictor.py` 89) — for a recall-weighted threshold a
silent 0.5 would flip many readmit decisions to negative with no error anywhere.

**The serving routine.** `ReadmissionPredictor.load` downloads the bundle, verifies
the digests, and reads the feature order, the parent groups and the threshold out of
it (68–92). `preprocess` rejects unknown feature keys, wrong-length vectors,
non-finite values and oversized batches (94–153). `postprocess` aggregates the
TreeSHAP contributions to parent groups, takes the ten largest by magnitude, and
returns the probability, the threshold decision, the base value, the declared
attribution units and the factors (161–192). One forward pass produces both the
probability and the attributions. The probability is the booster's own logistic
output and the pipeline protects it rather than correcting it: HPO fixes
`scale_pos_weight` at 1.0 for that reason (`optuna_hpo.py` 55) and the final fit pins
`objective=binary:logistic` (`train_final.py` 84–88), so the endpoint reports an
unweighted probability rather than a margin. No post-hoc calibration is applied —
there is no Platt or isotonic fit anywhere in the pipeline — and the reliability
curve that `evaluate_test` computes and plots (`evaluate_test.py` 137, 214) is a
diagnostic that no gate reads. The operating threshold is chosen separately, by the
calibrate-threshold step, from out-of-fold train probabilities.

**The registry entry and the deployment.** The pipeline records a model entry named
`readmission-final-<ts>` whose `artifact_uri` is the bundle, labelled with the test
AUCPR, the threshold, the run that produced it, the revision of the code and the row
count of the data (`provenance_labels`, `register_model.py` 127–160), with the same facts spelled out in
its description for a reader. Deployment is a separate
step and a *different* record: `deploy_cpr.py` discovers the newest such entry
(186–214), uploads its own model entry named `readmission-cpr-<ts>` carrying the CPR
container spec — by digest, not by tag (`267`–283) — deploys it to the endpoint at 0%
traffic while the old one still
serves, shifts traffic, asks the new deployment to answer as itself, and only then
undeploys the previous model (320, 346, 351, 367). Because
only the pipeline writes that name, and only a run whose gates passed registers, the
newest entry is the most recent successful run: the last deployment (2026-09-16)
resolved to `readmission-final-20260902014308`, from the training job of 2026-09-02,
with a test AUCPR of 0.3282 and a threshold of 0.11 — the values `mlops/README.md`
63–64 documents (read from the project's registry and pipeline job list, 2026-09-17).
The image tag is a hash over the Dockerfile, the predictor and the requirements
(67–73), so changed source means a new tag — though not a reproducible image, because
the build resolves `apt` and pip ranges when it runs, and the digest it looks up
(80–83) is discarded rather than recorded.

A deployment is not finished until the model answers. After the shift the script asks
the endpoint about one instance — every value null, keyed by the code-owned feature
contract — and requires the answer to have come from the deployment it just made;
if it did not, the previous traffic split is restored and the script exits non-zero
(`deploy_check.py` 139–180, called at `deploy_cpr.py` 351–357). That is why the
previous deployment is retired last: the thing to roll back to has to still be there.
It is the only step that can tell a working deploy from one that loads and serves
nothing, and it is deliberately narrow — whether the numbers are clinically sensible
is what the demo tests.

**How the application reaches it.** The tool builds a named instance in the order the
newest bundle's manifest declares (`services/mcp/dependencies/features/manifest.py`
81, `…/features/base.py` 47), calls the endpoint
(`services/mcp/dependencies/model_endpoint.py` 29–45), and reports the identity the
endpoint gives back with the answer — the deployment stamps the provenance record's
name into the container (`deploy_cpr.py` 267–283) and the CPR returns it on every
prediction (`predictor.py` 77, 205). No registry lookup is involved in naming the
model any more. The manifest lookup is still cached for the life of the process
(`manifest.py` 26), but it now decides only where the feature order comes from — a
question it can actually answer — and no longer doubles as an identity.

---

## 4. Current state against the requirement

| Requirement | State | Why |
|---|---|---|
| E5 — train → evaluate → registry → endpoint, lineage in ML Metadata | **Met** | The chain exists and runs: a pipeline that fits the model, four gates that fail the run rather than log and whose thresholds are code or a versioned artifact rather than inputs, a registry entry naming the run, the revision and the data, and an endpoint reached by one guarded deploy script. The lineage is in ML Metadata: the dataset artifact is an input of the load step and the registered model is an output of the registration step, both inside the run's own context. Verified by a completed run on 2026-09-17. |
| E6 — retraining triggered by monitoring | **Not met** | There is no monitoring job, no schedule, no topic and no trigger anywhere in the repository; retraining is a person running the pipeline. The operations document describes the loop as if it ran (gap 11). E6 is a SHOULD, so this is a product decision to take rather than a defect to repair — but the requirement is not met, and the document will not be the thing that changes it. |
| E1 — version control for weights, datasets and images | **Not met** | The weights are versioned the right way: one bundle per run in GCS, digests verified at load against a corrupt or half-written file, a non-executable format, and the gate numbers and the threshold travelling with them. The dataset has a recorded reference and a lineage artifact (gap 6), the repository holds no copy of the model to drift from it (gap 8), and the training image is named by build tag on every submission (gap 1). The images are the remaining failure: the CPR image is tagged by a hash of its source, but the build also publishes `:latest`, the build is not reproducible, and the digest it resolves is discarded rather than recorded (gap 12). |

---

## 5. Gaps

One block per gap: the defect and its evidence, the decision that closes it, and — once
it lands — what actually changed and how it was checked. A gap marked *Closed* has that
outcome recorded; the rest are open. Ordered by how much of the requirement each one
breaks. Twelve gaps: the first seven were found in the audit, the eighth and ninth were
with it, and gap 12 was found while closing gap 3 and is recorded here rather than
folded into it. Eleven of the twelve are closed, and gap 11 records its options and awaits a
decision. Line numbers are the working tree's, and a
number marked *then* is where the code sat when the gap was recorded — the change closed
it and moved the line.

**1 — The pipeline cannot be run from the repository as it stands.** *Closed
2026-09-17.*
Every training component imports its helper from project source at run time, so the
pipeline needs a container with that source baked in — stated in the Dockerfile's own
comment (`mlops/training/Dockerfile` 26–33) and enforced by `submit_pipeline.sh`, which
exports `TRAINING_IMAGE_URI` and says why (`submit_pipeline.sh` 38–51). That image could not be produced from what was committed. The prerequisite the
submit script named did not exist (`mlops/scripts/build_images.sh all`); the script is
`mlops/training/build_images.sh`, and it accepted only `training`. The build script
submitted `pipelines/cloudbuild.yaml` while the config is
`mlops/training/cloudbuild.yaml`, which built `-f pipelines/Dockerfile` while the file
is `mlops/training/Dockerfile` — and that Dockerfile baked the layout from before the
rename, `COPY src` and `COPY pipelines`, while the pipeline compiled beside it imported
`mlops.training.components.*`. The submit step then ran a path that did not exist:
`$MLOPS_DIR/pipelines/training_pipeline.py submit`. The package's own prose still sent
readers to the old names (`_image.py` 7, `encoding.py` 30), which is how a rename leaves
work behind after the paths are fixed.

Two committed copies of the IR sat side by side and nothing tied either to the source.
The copy at the repository root came from the pre-rename tree: it imported
`pipelines.components.*` and pinned the mutable `training:latest`. Its task list, its
components and every parameter default were in fact identical to what the source
compiles to — which is exactly what made the drift invisible, because nothing compared
them. The pipeline test would not have noticed a step going missing either: it asserted
that its expectation was a subset of the tasks wired, not equal to them. So the gates
this layer is audited against could not be re-run from the repository, and the model in
production was produced by a layout that no longer exists here.

Fix. Correct the paths and prove it by running it: the build script to the real config
and target, that config to the real Dockerfile, the Dockerfile to the layout after the
rename, the submit step to the real module path. Single-source the IR beside the source
that compiles it, assert the task list by equality rather than containment, and run the
pipeline once — a DAG nobody can execute is a description, not a deployable artifact.

Change. Every path names what exists, the root IR is deleted, and `compile_pipeline()`
writes the only copy beside its source. The guard test compares the committed IR against
a fresh compile as parsed data — formatting may differ, a changed task, component or
default may not — and asserts the run names exactly one image; the task list is an
equality against the eleven expected names (`mlops/training/tests/test_pipeline.py` 30,
46). The proof is a run: `readmission-training-20260917130517`, 2026-09-17 17:05 UTC,
`n_trials=5`, build-tagged image, SUCCEEDED with all eleven tasks green. Its lineage is
real rather than implied — `register-model`'s execution consumed `booster_model` from
`train-final` and `manifest` from `load-data` and produced a `serving_model` artifact,
inside the run's own context — and `benchmark-gate` sits between `benchmark-xgboost`
and `optuna-hpo`, so a benchmark below the baseline stops the run before any training.
The submission path cannot record a mutable image either: `submit_pipeline.sh` stops
when `TRAINING_IMAGE_URI` is unset and prints the available builds, so every run names
the image that produced it. The verifying run was a five-trial wiring check — the
pipeline it exercised is the one that ships.

**2 — The endpoint does not identify the model, and the caller discards the identity
it is given.** *Closed 2026-09-17*
Vertex returns a deployed-model id with every prediction — the smoke test prints it
(`smoke_test.py` 204) — and the tool's endpoint wrapper returns only the first
prediction, dropping it (`model_endpoint.py` 39). The name the user sees comes from
somewhere else: the tool filled `model_version` from the newest
`readmission-final-*` registry entry — the lookup lived in
`...features.manifest.model_version()`, since deleted — and cached it for the life of
the process (`manifest.py` 26), while the endpoint serves whatever bundle was
deployed when the deploy script last ran (`deploy_cpr.py` 346). Nothing compared
them. Three consequences, all silent. The answer was attributed to a model that may
not have produced it — the defect layer 4 recorded as "recording the model we asked
for, not the one that served". Two instances serving the same traffic could disagree
about which model answered, because one resolved the cache before a retrain and the
other after. And the same cached lookup supplies the order the vector is built in
(`manifest.py` 81), which the wrapper then sent positionally (`base.py` 42), so a
retrain that reordered the features sent values in one order to a booster that read
them in another: equal-width vectors scored the wrong patient with no error at all.
The endpoint would reject a wrong length (`predictor.py` 141–145) and does reject unknown
*names* (`predictor.py` 122–128) — but the tool never sent names.

Fix. Take the identity from the party that knows it: keep the `deployed_model_id` Vertex
already returns, stamp the deployed record's name into the container as `MODEL_VERSION`,
and delete the registry lookup that answered the question with the wrong model. Send the
instance keyed by feature name, so the endpoint's existing unknown-name refusal becomes
reachable, and give the endpoint the mirror guard it lacks — an instance that omits a
declared feature is refused rather than scored as if that measurement were absent.

Change. The payload carries the deployed-model id (`model_endpoint.py`), the stamp is
read at load (`predictor.py` 77) and returned with every prediction (205), and the tool
reports the endpoint's name with the id as its fallback. The instance is keyed by name
(`base.py` 47); a partial instance is refused (`predictor.py` 134–139) while a null value
still passes. The `model_version()` registry lookup is deleted and a test fails if it
returns. Six tests (`tests/agent/test_predict_identity.py`,
`mlops/training/tests/test_gate_integrity.py`), plus the live call the end-of-day
deploy makes. The tool still reads the feature order from the newest bundle rather
than from the deployment, so a retrain the deployment has not caught up with fails the
call loudly instead of answering wrongly — that is the intended trade, because a
refusal is a fact and a wrong number is not.

**3 — The deployment step is unguarded, and one of its two paths deploys the
provenance record itself.** *Closed 2026-09-17*
`deploy_cpr.py` shifted all the traffic to the new deployment and retired the
previous one without ever asking the new model a question — the deploy, the shift
and the retiral were consecutive statements with nothing between them. The 0% stage
proved that the container became ready, not that the model answered, so a bundle that
loaded and scored wrongly — which is what gap 4's hand re-registration produces —
would have served until somebody used the demo.

A second script in the same folder deployed a *provenance record* rather than a CPR
model: the newest `readmission-final-*` entry, whose serving image was the mutable
`readmission-cpr:latest` (`register_model.py` 215 then), straight to full traffic with
no 0% stage, no retiral of the previous deployment, no predict or health route, and
without the `GOOGLE_CLOUD_PROJECT` environment that `deploy_cpr.py` documents as the
fix for a worker that never becomes ready (196–198). It was named in no runbook or
script list — the only way to find it was to list the folder — and its
`--include-models` flag deleted every `readmission-final-*` record, the namespace the
running application resolves its feature order from. (The script was
`mlops/serving/deploy_endpoint.py`; it no longer exists.)

The one tool that could have checked a live endpoint, `smoke_test.py`, was called by
neither path and did not run as committed: `ModuleNotFoundError: No module named
'mlops'`, because its `sys.path` insertion resolved to `mlops/` rather than the
repository root (28), and its fixture path pointed at a `projects/` directory this
repository does not have (41–42).

Fix. One deploy path: delete the second script, whose teardown `scripts/agent/teardown.py`
already covers, and covers better. Then end the deploy with a question rather than a
print — after the traffic shift, ask the new deployment one instance and require both
identities, Vertex's deployed-model id and the endpoint's own stamp. On any failure
restore the previous traffic split and exit non-zero, which is what forces retiral to be
the last step. Retry everything, identity included, because a fresh deployment is
legitimately slow to serve.

Change. `mlops/serving/deploy_check.py` holds the check and `deploy_cpr.py` calls it
after the shift (260–266); `mlops/serving/deploy_endpoint.py` is deleted. The probe
instance is keyed by the code-owned feature contract and carries nulls, so it needs no
patient and doubles as a check that the deployed bundle's manifest agrees with the code.
Sixteen tests (`mlops/training/tests/test_deploy_verification.py`) pin the response
contract, both identities, the retries and the rollback — the last asserted against a
fake endpoint that records its calls. Proven by those tests; the end-of-day deploy is
the first real call the check makes. `smoke_test.py` is left as it was found, because
nothing depends on it now.

**4 — Nothing stops a second writer from taking the newest position, and nothing can
say which of two successful runs was better.** *Closed 2026-09-17*
The pipeline uploads `readmission-final-<ts>`, labelled with the test AUCPR and the
threshold (`provenance_labels`, `register_model.py` 127–160). Every selection in the system is then one
query — newest by `create_time` on that prefix: `deploy_cpr.py` 186–214,
`manifest.py` 47–50, `smoke_test.py` 53–64 — and that
query is equivalent to "the most recent successful run" only while the pipeline is
the only writer. It is: registration is wired after the eval gate and both audits
(`training_pipeline.py` 257), so a record exists only for a run whose gates passed,
and every such run writes exactly one. The equivalence breaks at the second writer.
`register_serving_model.py` uploads the same name for a bundle that already exists,
with no gate metrics and no digests of its own (64–73), so a hand re-registration
takes the newest position without training anything and becomes the bundle the
application reads its feature order from and names behind a score. That path is
covered by no test, and it did not even run as committed — its path insertion pointed
at the package instead of the repository root (`register_serving_model.py` 37).
The supporting half is what happens when two runs both succeed: the registry holds
five `final` records — 2026-07-23 (no metric labels), two on 2026-09-01 scoring
0.4097 and 0.4083 at threshold 0.12, 2026-09-02 scoring 0.3282 at 0.11, and
2026-09-17 scoring 0.3294 at 0.11 — and no code in `mlops/`,
`services/` or `scripts/` reads those labels, so nothing in the system can express
"this one should serve instead". The two higher-scoring records are not comparable to
the served one (a different cohort), which is exactly the judgement the code cannot
make and a stored metric could.

Fix. State the rule and keep it true: the model that serves is the bundle from the most
recent successful pipeline run. That holds today because the pipeline is the only writer
and only a successful run registers, so the rule itself does not change — the work is to
give the manual path its own namespace so it cannot take the newest position without
training, and to make the resolution say what it is doing instead of leaning on creation
order, which needs the run identity from gap 5 to be checkable. The metric labels stay
for the case they exist for: two runs that both succeed. Test: a hand-registered record
is not what the deploy script or the application resolves.

Change. The rule is now stated once, as a pure function — `select_serving_record` — and
what decides is the label, not the name: a record serves only if the training pipeline
registered it, meaning the `readmission-final-*` prefix *and* the `stage=final` label
`register_model.py` writes with it. The manual path publishes `readmission-manual-*` with
`stage=manual`, so no resolver can select it; its path insertion, which pointed at the
package instead of the repository root and stopped it importing anything, is fixed, and
it now prints that it will not be selected. All three resolvers apply the same predicate
— `deploy_cpr.py` 99–116, the tool's `manifest.py` 47–50, `smoke_test.py` 58 — and a test fails
if any of them starts selecting on the name alone. The deplorer prints the record it
chose with the score its labels carry (`test AUCPR 0.3294, threshold 0.11`) and the
previous registration beside it, which is the first time anything reads those labels:
they were written and never consulted. Thirteen tests.
The service cannot import the function — `services/mcp/` does not depend on `mlops/` —
so the contract between the one writer and the three readers is the label value itself,
which is written in one place and named by all four.

**Found live while closing this, and it is the defect happening:** the newest `final`
record is `readmission-final-20260917173317` — the five-trial run that verified gap 1
earlier the same day, at test AUCPR 0.3294 — so the next deploy would serve a model
whose registration came from a pipeline test rather than the fifty-trial fit the endpoint
currently serves. Creation order cannot fix that and the rule cannot either: the five-trial
run really did register, and really did pass the gates. Either the record goes, or the
deploy pins its bundle deliberately.

**5 — The registry entry records no lineage: no version, no run, no code.** *Closed
2026-09-17*
What the entry carried was a bundle, an image, a name with a timestamp in it and four
labels — the pipeline, the stage, the test AUCPR and the threshold
(`provenance_labels` 127–160). What it did not carry is which run produced it: the
the pipeline job name was passed to five other components (`training_pipeline.py`
176, 189, 204, 220, 235) and not to registration, and no revision of the code was recorded
anywhere. `Model.upload` was also handed a `parent_model` with a comment claiming
version lineage, and nothing ever supplied one — the component parameter, the wrapper
and the pipeline parameter all defaulted to empty — so every run created a sibling
record and the relationship between two models was a timestamp inside a display name.
Whether Vertex links a model to its run through ML Metadata was not something this
repository arranged or checked; it does, and the gap-1 verification run shows it — the
`serving_model` artifact is an output of that run's register-model execution, inside
the run's own context — but that is the artifact, not the registry entry a person looks
up, and nothing here read either. (The dead `parent_model` parameter has since been
deleted; there is no version lineage, which is now visible rather than implied.)

Fix. Record *which run* and *which commit*: pass the job name that is already in scope
into registration, resolve the revision once at submit time, and put both on the entry.
Then either set `parent_model` from the previous registration so versions exist — which
obliges re-testing every discovery query against the versioned shape — or delete the
parameter and the comment that claims it. A lineage parameter nothing sets reads as a
capability this layer does not have.

Change. Registration records the run and the code on the entry itself.
`provenance_labels` (`register_model.py` 127–160) writes `run` and `commit` beside the
score labels, and the description spells both out for a reader in the console: run,
revision, test AUCPR and threshold in one line. The job name was already in scope one
line above the call and is now passed to it (`training_pipeline.py` 255); the revision
is resolved once at submit time — `git rev-parse --short HEAD`, with `-dirty` appended
when the worktree is modified, in `submit_pipeline.sh` — and sent as a parameter value
(`training_pipeline.py` 309) rather than compiled in, because a revision baked into the
pipeline definition would describe the machine that compiled it, not the run. An unknown
value is left off the entry instead of written as a placeholder, so an absent label means
not recorded. The dead `parent_model` is deleted from the component, the wrapper, the
pipeline signature and the upload: no version lineage pretends to exist. Ten tests — the
encoding is registry-legal, a long or punctuated value is sanitised and truncated, empty
values are omitted, the parameter cannot come back, and the committed IR bakes no
revision.

**6 — The model does not say what it was trained on.** *Closed 2026-09-17*
The registry entry's labels are the pipeline, the stage, the run, the revision, the
test AUCPR and the threshold (`provenance_labels`, `register_model.py` 127–160); its `artifact_uri` is the
bundle (187); and the bundle holds the booster, the manifest, the threshold, the gate
metrics and the digests (44–85, 178–179) — with no reference to the data behind any of
them. The table it read is a parameter (`training_pipeline.py` 99, 297) resolving to the
Dataform view `analytics_dataset_encoded` (`mlops/data/config.py` 66), which Dataform
rebuilds in place — it is a table of 219,744 rows, last written 2026-09-02 01:08 UTC, and
the run that trained on it began 35 minutes later. There is no dataset version to cite
and no pointer to one: two models registered a month apart are indistinguishable in what
they learned from, so a question about a surprising score — was the data wrong, or the
model? — cannot be answered from the artifacts.

Fix. Record *which data*, and put it where the artifacts are. The pipeline already
knows what it read — the table reference is a parameter and the row counts are in hand —
so the reference should be captured as the data is read, carried in the manifest that
ships inside the bundle, and surfaced on the registry entry. What it names should be the
state that was observed rather than the table's name: the table is rebuilt in place, so
a name is not a version, and a reference that can silently come to mean different bytes
is worse than none.

Change. One fact, three places, none of them inferred later. `load_data` asks the table
for its own metadata at the moment it reads — row count, last written, and the rows it
saw per split — before the data can move (`_data_reference`, `load_data.py` 83–105), and
writes that into the manifest (`load_data.py` 209), so the reference ships inside the
bundle and is covered by its checksums. The table is also a node in the lineage now: an
importer names it once as `bq://<project.dataset.table>` and `load-data` consumes the
resulting `system.Dataset` artifact instead of taking a second parameter that could
disagree with it (`training_pipeline.py` 132–142). Registration reads the reference back
out of the manifest it already copied (`register_model.py` 243–244) and puts the row count on
the entry as a label beside the run and the revision (`provenance_labels` 127–160). Ten tests. The DAG is
twelve tasks rather than eleven: an importer is metadata-only, so nothing new runs and
nothing new is billed.

Two limits, taken deliberately. It is not a content digest: a rebuild that
changed values and kept the row count looks identical to this reference. And the artifact
URI points at a table rebuilt in place, so it names the state that was observed, not bytes
that cannot move — `CREATE SNAPSHOT TABLE` or a `FOR SYSTEM_TIME AS OF` reference is what
would add re-readability, at the cost of storage or a retention window. Neither limit
stops the requirement being met: the model names its data, and the reference is observed
rather than inferred.

**7 — The gate baselines are overridable at submit time.** *Closed 2026-09-17*
`hospital_aucpr` and `max_drifted_share` are pipeline parameters with defaults
(`training_pipeline.py` 111–112), so they are answered at submit time, and
`validate_baseline` refuses only a value outside (0, 1) (`benchmark_gate.py` 20–28). A
submitter passing `0.01` neutralises the benchmark gate and the test gate together
(`evaluate_test.py` 83 uses the same value), and the tests probe only the implausible
values (`mlops/training/tests/test_gates.py` 54–61). The gate metrics written into
the bundle would then record, faithfully, that a model which beat nothing had passed.
This is the residue of the ECC-65 finding: the build-time default is real, and the
override is still a parameter. The reason the parameter existed is the defect worth
naming: the baseline artifact is kept **out** of the training image
(`mlops/.gcloudignore` 11–12 ships only the missingness policy), so a gate container
cannot read its own threshold and the value had to be resolved on the submitting
machine and carried in. The gate was handed the number it was supposed to enforce.

Fix. Put the baseline where the gate reads it: load it from the versioned artifact inside
`benchmark_gate` and `evaluate_test` and delete the parameter, or pin it as a code
constant. The drift share is a judgement call, so pinning it with a documented value is
the honest form of "a setting". Test: a run submitted with a hostile `hospital_aucpr`
cannot change the verdict.

Change. Neither threshold is an input any more, so there is nothing to pass — a
stronger guarantee than validating a value, because it cannot be argued with at
submit time. The drift share is a constant with its reasoning beside it
(`_baselines.py` 28–33) and the gate reads that (`validate_data.py` 60); the baseline
is read from the artifact **inside** the gate container (`_baselines.py` 36–57,
`benchmark_gate.py` 16) — which is why `artifacts/hospital_baseline.json` was added to
the build context (`mlops/.gcloudignore` 14): the only copy a gate will read is the one
that travels in the image the run records, so the run's image tag names the baseline
version that gated it. `validate_baseline` survives as the artifact's guard rather than
the parameter's (`benchmark_gate.py` 23): a missing or corrupt baseline stops the gate
instead of gating against a number nobody can account for. `HOSPITAL_AUCPR` and the
build-time resolver are deleted from the pipeline, and `submit()` never had to be
trusted with either value again. Three tests, including one that asserts neither name
appears in the compiled IR or on any gate task, and one that the artifact is on the
build path.

**8 — The repository held a mixture: July's booster with September's threshold.**
*Closed 2026-09-17*
`model.bst`, `manifest.json` and `threshold.json` sat at the repository root, where the
scripts that build the demo cohort read them (`fill_features.py` 35–38, 390 then,
imported by the cohort builders; and `generate_synthetic_cohort.py`, since deleted).
Compared by hash
against every registered bundle, `model.bst` is byte-identical to the booster of
`readmission-final-20260723172647` (2026-07-23), whose own `threshold.json` says
0.12 — while the `threshold.json` committed beside it said 0.11, the value in the
2026-09-02 bundle. The pair came from two different trainings and never existed as a
registered model. `model.bst` was not even committed (`.gitignore` 45), so a clean
checkout held a threshold for a model it did not contain. `generate_synthetic_cohort.py`
hardcoded `0.12` with the comment "from threshold.json" while nothing imported it at
all — a false comment on dead code.

The consequence was measurable. The fixture records `"threshold": 0.12` on
every patient, because it was built while a 0.12 bundle was current; the endpoint serves
the 2026-09-02 bundle at 0.11. Ten of the 108 patients in
`data/agent/demo_fixtures/cohort_risk.json` have probabilities between 0.11 and 0.12,
so the fixture calls them negative where the live path calls them positive: the offline
fixture and the live system disagreed about ten of 108 patients, on opposite decisions.
It is a local artifact rather than a committed one — `data/` is gitignored
(`.gitignore` 33), so it is present on the machine that built it and absent from a clean
checkout, which is how a threshold for a booster `model.bst` was never in the repository
either.

Fix. Fetch the booster from the registered bundle rather than reading an untracked file
on one machine — or drop it, and let the scripts fail when it is absent. Read the
threshold from the file in both scripts instead of restating it as a literal. Label the
cohort artifact with the run that produced it. Test: a working copy's booster and
threshold came from the same bundle, which is the assertion that would have caught this
swap the day it happened.

Change. The loose copies are gone — `model.bst`, `manifest.json`,
`threshold.json` — and with them the only reason they could disagree. `fill_features.py`
now resolves a bundle and downloads all three files from it into one directory
(`_serving_bundle`, `fill_features.py` 41–86), using the same rule the deploy script
uses to choose what serves (`select_serving_record`), so the cohort is scored by the
model that answers in production rather than by whichever copy happened to be on disk.
`BUNDLE_URI` still pins a specific run. The three files are read from that one
directory (`fill_features.py` 89–94, and the booster at 446), and the cache directory is keyed by a hash of the
bundle URI, because a shared directory would put one bundle's manifest beside another's
booster and rebuild the same defect one run later. `generate_synthetic_cohort.py` is
deleted: nothing imported it and its comment was untrue. Eight tests
(`tests/agent/test_serving_bundle_source.py`), including one per file asserting no loose
copy exists at the root and none is read from there again.

**Still true, and it is a different problem:** the fixture is stale. Rebuilding
it is gap 8's follow-through, not its closure — the mixture is gone, and the fixture now
has to be regenerated from the bundle that serves. Whoever regenerates it should read gap
11 first: the `readmission_30d` column the demo cohort carries is derived from the model's
own scores, so it can be used to rebuild the fixture and must not be used to report
accuracy.

**9 — The feature contract is restated five times, and a divergence would arrive as
a missing value rather than an error.** *Closed 2026-09-17*
The 49 names the served rows are built from were written out by hand in four places,
and they agreed — compared during this audit, name for name and in order: the
training contract (`mlops/data/encoding.py` 169), and
three demo scripts (`generate_hybrid_features_v2.py` 56 then, `load_hybrid_notes.py` 40
then, `load_synthetic_features.py` 35 then). The values are not re-encoded either: those scripts
read them out of the training view by name (`generate_hybrid_features_v2.py` 50,
230), so the one-hot semantics are shared and there is no encoder skew. What was not
shared is ownership, and the failure mode was silent. The read path filled every column
the row lacked with `None` (`bigquery_source.py` 45 then), so a name that drifted out of
one of the five lists arrived at the model as null — which the booster reads as
"missing" and scores confidently, rather than raising. The guard that would catch it
was unreachable on this path: the tool asks whether any expected column is absent
(`prediction.py` 77) and reports `incomplete_features`, but this source has already
put every name into the row, so only a test-built row can trigger it
(`test_error_disclosure.py` 205).

Fix. The list is the model's contract, so put it in one place the scripts import instead
of restating — the same remedy layer 5 applied to the embedding parameters, for the same
reason. Then stop the read path collapsing two different things into `None`: a null
*value* is a legitimate missing measurement, an absent *column* is a defect, and
reporting it as the latter lets the tool's `incomplete_features` path fire — a check
already written, already tested, and currently unreachable.

Change. The contract has one owner. The three demo generators import `feature_order()`
from the training contract (`mlops/data/encoding.py` 169) instead of re-typing the 49
names — `load_synthetic_features.py` 42, `load_hybrid_notes.py` 43,
`generate_hybrid_features_v2.py` 59 — so what the audit checked by eye is now the only
thing they can do: a test parses each file and fails if the assignment is a literal list
rather than a call to the contract, and a second test pins the contract at 49 names in
the generators and the model alike.

The read path stops conflating two different things. `bigquery_source.fetch` keeps a
column whose *value* is null and leaves out a column the table does not *carry* — the
row is filtered to the contract's names and nothing is invented to fill a gap
(`bigquery_source.py` 55–60). A null value still reaches the booster and is read as NaN,
which is what the model was trained for; an absent column now arrives absent, so the
tool's guard fires and the request is refused with `incomplete_features` instead of being
scored. That guard was written, commented and tested (`prediction.py` 95–107,
`test_error_disclosure.py` 205) and had never run on the live path, because the source it
read from made it impossible to fail.

Nine tests (`tests/agent/test_feature_contract.py`): each generator derives the contract
rather than restating it, each generator's contract is the 49 the model was trained on,
the label never reaches the model input, a null value keeps its column, an absent column
is left out so the row is refused. The whole suite is 522 passed, 6 skipped, with the ten
pre-existing environment errors.

**Found live while closing this, and it is layer 5's, not this layer's:**
`services/mcp/dependencies/features/bigquery_source.py` imported its config with
`from ..config import ...` — one dot short of `services/mcp/config.py` — so the module
raised `ModuleNotFoundError` on import and `get_feature_source()` could not build. Every
prediction and every retrieval would have returned "the feature source could not be read".
It is in `HEAD`, and no test imported the module, which is why a green suite hid it: the
defect was found only because closing gap 9 required executing the file that had it. The
import is `from ...config import ...` and the source builds again (`BigQueryFeatureSource
| table: readmission.hybrid_features`), but the missing coverage and the owner of this
path — layer 5 — are recorded here rather than fixed here, and layer 5's document should
carry it.

**10 — The gate metrics are written after the digests meant to guard them.** *Closed
2026-09-17*
`assemble_serving_bundle` computes SHA-256 over `model.bst`, `manifest.json`,
`threshold.json` and `gate_metrics.json`, and writes `checksums.json`
(`register_model.py` 76–84 then). It walked that list and skipped any name that did not
exist yet (80 then), and `run_register_model` called it first (110 then) and wrote
`gate_metrics.json` afterwards (128–129 then). In the fresh artifact directory a pipeline
run uses, the file does not exist when the digests are taken, so it is skipped — the
four numbers that say why this model was allowed to register are the one part of the
bundle the endpoint does not verify (`predictor.py` 40–66 verifies what
`checksums.json` names). Confirmed on the bundle that currently serves:
`.../readmission-training-20260917130517/register-model_5840845322842013696/serving_model`
holds four files and its `checksums.json` names three, and nothing in the repository
reads `gate_metrics.json` at all.

Fix. Move the `gate_metrics.json` write ahead of the digest pass so the file is covered
by the same manifest as everything else, or stop naming it in the checksum list — a name
that can never resolve is a silent skip rather than a guarantee. The first is one line
and keeps the guarantee. Test: assemble a bundle in a fresh directory and assert every
file in it is named by `checksums.json`, which is the check the endpoint's own
verification cannot make.

Change. The digests are now derived from what was written rather than checked against a
list, which makes both halves of the defect — a file that escapes the digests, and a name
that cannot resolve — unrepresentable. `assemble_serving_bundle` appends each file to a
`written` list as it writes it and digests exactly that list
(`register_model.py` 104–107); the gate metrics are built in `run_register_model` (222–229)
and handed to the assembler so they are on disk before the digest pass, and the
`os.path.exists` line that swallowed the absent file is gone. The metrics block moved
above the digest pass; no other behaviour changed, and the order is commented where it
matters: the numbers that justify registration cannot ship unverified.

Six tests (`mlops/training/tests/test_bundle_integrity.py`): the gate metrics are covered,
the set named equals the set of files present, every digest is the SHA-256 of the bytes on
disk, an omitted threshold or an omitted metrics block produces no unresolved name, and an
AST check fails if the existence-skip returns. The live effect is one bundle file moving
from unverified to verified; the model, the threshold and the scoring path are untouched.
The fix reaches GCS with the next training image build, since a component's source is
baked into the image the run records — the bundle serving today still carries three
digests, and rebuilding the image is what changes that.

**11 — The monitoring that is documented as triggering retraining does not exist.**
*Decision pending 2026-09-17.*
`docs/operations.md` 44–45 states Vertex Model Monitoring on input and attribution
drift, a scheduled job that recomputes AUCPR on matured labels, and a Pub/Sub topic
that carries the retraining signal. No monitoring job, schedule, topic, subscription
or trigger exists anywhere in the repository or in the deploy scripts; the only
mention of model monitoring in the code is a phrase in `deploy_cpr.py`'s module
docstring (21). Retraining is a person running the pipeline
(`training_pipeline.py` 278–327). The environment script enables the Pub/Sub and
Scheduler APIs (`scripts/setup_environment.sh` 24–25) and grants the pipeline account
Pub/Sub editor (31) — the scaffolding for a loop rather than a loop. E6 is a SHOULD,
so this is a decision to make rather than a defect to fix, but the operations
document currently describes the loop as running.

Fix. E6 is a SHOULD, so this is a product decision and the honest default is the second
option: correct `docs/operations.md` to say retraining is manual and monitoring is
planned, and say what would trigger it. Building the loop means a monitoring job, a
schedule, a signal and something that acts on it — real work, worth doing deliberately
rather than announcing first. What is not defensible is a document describing a mechanism
that does not run.

*Decision pending.* The options were worked out on 2026-09-17 and are recorded here with
the constraint that shapes them, because the constraint is not obvious and it rules out the
metric a monitoring story would normally lead with.

**The demo cannot supply the label that performance monitoring needs — and it appears to.**
The served cohort's table carries `readmission_30d` (89 rows, 13 positive), but it is not an
outcome: `generate_hybrid_features_v2.py` 281–286 ranks the rows by the model's own
probability and labels the top 15% positive. The label is therefore a function of the score
it would be used to judge, so an AUCPR computed against it is circular and would look near
perfect. Nothing may report it as model quality. Real performance monitoring needs labels
that mature 30 days after discharge, which no demo cohort has.

*What is observable without labels, in descending value:*

1. **Serving-path health.** The tool already returns and logs structured codes —
   `feature_fetch_failed`, `incomplete_features`, `model_unavailable`,
   `invalid_tool_response` (`prediction.py` 43–54, 76–165) — so a log-based metric and an
   alert policy are configuration rather than code. This is the signal that would have
   caught the unimportable feature source on 2026-09-17 (see gap 9) instead of a person
   finding it.
2. **A prediction log.** One row per call: admission, probability, decision, threshold,
   `deployed_model_id`, feature source, error code, timestamp. Nothing writes one. The
   `readmission.test_predictions` table (49,103 rows, last written 2026-08-05 at threshold
   0.12) is an orphan: no code in the repository reads or writes it. The log is the
   substrate for the rest, and it makes "which model answered, at which threshold"
   auditable — layer 4's question, and this layer's evidence.
3. **Drift, which needs no labels at all.** Input drift: the 49 features against the
   training distribution. Prediction drift: the score distribution and the decision rate
   against training. The mechanism already exists — `hospital_baseline.json` ships inside
   the training image and `validate_data.py` computes a drifted share against it — and
   serving-side monitoring is the same statistic applied to requests instead of the
   training table.

*Options.*

- **A — Correct the document only.** `docs/operations.md` 44–45 stops describing a loop
  that does not run, and says retraining is manual and what would trigger it. No code. E6
  stays *Not met*.
- **B — Correct the document, then build the three label-free signals** (serving-path
  alerting, prediction log, drift reports). E6 still stays *Not met*: detection without
  something that acts on it is half the requirement, and the honest split is that
  detection becomes real while the trigger does not. The reason for stopping there is
  stateable — a frozen 89-patient cohort gives drift nothing to move.
- **C — The full loop.** B plus Cloud Scheduler, a Pub/Sub topic and a job that submits
  the pipeline. Real work, and in the demo it can never fire.

Open question if B or C is chosen: the prediction log belongs either in the MCP predict
tool, which already holds the row, the score and the deployed model id, or in the endpoint
itself, where request logging is not currently configured (`deploy_cpr.py` has no logging
setup).

**12 — The image that serves cannot be pinned to immutable bytes, and a mutable tag is
published beside the pinned one.** *Closed 2026-09-17*
The CPR image tag is a hash over the Dockerfile, the predictor and the requirements
(`deploy_cpr.py` 119–126 then), so changed source means a changed tag. Two things weaken that
to less than the requirement asks for. The build published `:latest` alongside the
hashed tag (`deploy_cpr.py` 138–145 then, and the `:latest` tag in
`mlops/serving/cpr/cloudbuild.yaml`), so a mutable name pointed at the same bytes and
nothing stopped anything from deploying it — and it was not decorative: when no image URI
was passed, registration recorded `readmission-cpr:latest` on the entry
(`register_model.py` 215 then). And the digest the script resolved
(`image_exists`, `deploy_cpr.py` 128–135 then) was used as a boolean and discarded: the one
immutable identifier of the bytes that ran was fetched and then thrown away, so no
artifact recorded it. The image is also not bit-reproducible — the build resolves `apt`
and pip ranges when it runs — which is acceptable if it is stated, and was neither stated
nor compensated by a recorded digest.

Fix. Record the digest where the deployment is recorded, so the immutable identifier
of the serving image travels with the model the way the run and the revision now do.
Stop publishing `:latest`, or make plain that nothing may consume it. Then either pin
the build inputs or say in the runbook that images are reproducible by tag content and
not by bytes.

Change. An image recorded on an artifact is now recorded by digest, and the rule lives
in one place so the three call sites cannot drift: `mlops/serving/image_ref.py`
(`require_serving_image`, `digest_label`), used by the training registration, the manual
registration and the deploy.

The deploy resolves the digest after the build or reuse decision and returns
`repo@sha256:...` rather than a tag (`ensure_image`, `deploy_cpr.py` 159–183); what is
handed to `Model.upload` is that reference (267–283), so the serving-container spec — the
record the registry keeps — identifies the bytes. The digest is also written on the
deployment itself, full value in the description and truncated in an `image_digest` label,
so it can be read without asking Artifact Registry what it holds today. `resolve_digest`
replaced `image_exists` and returns the value it used to discard, and a build whose digest
cannot be read back stops the deploy instead of proceeding (172–180).

The mutable name is gone from every place it existed: the build no longer tags `:latest`
(`mlops/serving/cpr/cloudbuild.yaml`), the stale `latest` tag was deleted from Artifact
Registry, the `:latest` fallback is removed from registration — an empty
`serving_container_image_uri` now raises, and a tag is refused as well as an empty value —
and `test_cpr_local.py` resolves by digest too. The pipeline cannot know the digest, so
`submit_pipeline.sh` resolves the newest CPR image version at submit time and passes
`repo@sha256:...`; when no CPR image exists it stops and says how to build one, which is
gap 1's pattern applied to the second image this pipeline depends on.

Seventeen tests (`mlops/training/tests/test_image_identity.py`): no build step publishes a
mutable tag, the reference `ensure_image` returns is a digest, an unresolvable digest stops
the deploy, the upload argument is that reference, the digest label is registry-legal, an
empty or tagged `SERVING_IMAGE` is refused, the rule has one definition, the submit script
resolves a version and fails when there is none, and the committed IR bakes neither a tag
nor a resolved digest.

Verified against the real registry rather than only by test. The first build after the
change published one tag and no `latest`
(`1283d1649caa` → `sha256:4994534bf57ec7f0c470e6e6fbab4e56573aae281469090c389712266bc6359a`)
and the deploy path printed that digest as its reference
(`...readmission-cpr@sha256:4994534b…`); `gcloud artifacts docker tags list` then showed
three content-hash tags and no `latest`. The tag had pointed at older bytes than the
newest build, which is the defect in miniature: a name that tells you nothing about what
it resolves to.

The deployment that serves today was uploaded before this change, so its record names a
tag; the next deploy records a digest. And one item is recorded rather than fixed:
`test_cpr_local.py` also loads the predictor from `mlops/pipelines/serving/cpr`, a path
from before the rename, so the harness is not runnable and nothing depends on it — the
same class of leftover as `mlops/training/smoke_test.py`.

**Recorded, not owned here:** the tool contract and the validation of what reaches
the model are layer 5's; the feature view, the note tables and the retrieval index
are layer 7's, including the ingest pipeline that builds the index and the demo rows
that feed the served path; the execution record and lineage as something you can
query are layer 10's; the evaluation harness that would judge a retrained model is
layer 9's; image scanning, provenance attestation, "only approved images deploy" and
the project id committed in five scripts are layer 11's; and the front door that
makes the endpoint private is layer 2's.

---

## 6. Interview questions this layer answers

**Where does the risk score come from?**
A model we trained ourselves: gradient-boosted trees fitted on MIMIC-IV admissions,
predicting 30-day readmission at discharge. It is served behind a Vertex AI endpoint
by our own prediction container, which returns a probability, a threshold decision
and exact TreeSHAP attributions in one pass. The probability is the booster's own
logistic output on the probability scale, not a post-hoc calibrated one: the training
fixes `scale_pos_weight` and pins a logistic objective to keep it that way, and no
Platt or isotonic fit is applied. The reliability curve the evaluation computes is a
diagnostic nobody gates on. The operating threshold is a separate decision — chosen on
out-of-fold train probabilities by the calibrate-threshold step, and shipped in the
bundle as metadata rather than baked into the model.

**How does a new model reach production?**
A pipeline run fits it, and it can only be published if it passes four gates: the
drift gate, the benchmark gate, the held-out test gate with a stability check, and
the ordering that keeps registration behind the audits. Registration publishes a
bundle — booster, feature manifest, threshold, gate metrics, digests — plus a
registry entry. Deployment is a separate, deliberate step that uploads a second
record, deploys it at 0% traffic while the old model keeps serving, shifts traffic,
asks the new deployment to answer as itself, and only then retires the old one; if
that answer does not come back from the model just deployed, the traffic is put back
where it was and the command fails. The model it deploys is the bundle from the most
recent successful run, which the deploy script resolves by requiring both the
pipeline's name prefix and the stage label it writes with it, so a hand-registered
record cannot take that position. The pipeline can be re-run from the repository —
the last run, `readmission-training-20260917130517`, finished with every task green —
and its lineage runs from the dataset artifact through the audits to the registered
model.

**What stops a bad model from being served?**
The gates fail the run, not the request: a model that does not beat the baseline by a
margin, or that degrades too far from its validation estimate, raises inside the
pipeline and never reaches registration. The endpoint then refuses to start at all if
the bundle's digests do not match or if the threshold is missing — a bundle that
cannot be verified is never served, and a threshold is never defaulted. The deploy
checks it too: the previous model is not retired until the new one has answered a
question as itself, so a deploy that produces a dead endpoint fails as a command
rather than as a demo. And no gate threshold is an input any longer: the baseline is
read inside each gate from the artifact baked into the image the run records, and the
drift share is a constant in the code, so there is nothing for a submitter to set —
weakening a gate now means changing the code or the artifact in a commit, where it is
reviewable.

**If I retrain and forget to redeploy, what happens?**
The answer is still attributed correctly — the name comes from the endpoint, so it
reports the model that actually ran rather than the newest one registered. The other
half is now a refusal instead of a wrong number: the tool sends features by name, so
a retrain that adds or renames one makes the call fail loudly rather than scoring a
vector against a vocabulary the endpoint does not share. What remains is the mismatch
itself — the tool reads the feature order from the newest bundle while the endpoint
serves an older one — so until the deployment catches up, questions fail rather than
being answered wrongly.

**Which model answered that question — can you prove it?**
Yes, and it costs nothing: Vertex returns the deployed model's id on every response,
the deployment stamps the provenance record's name into the container, and the answer
reports it. Until 2026-09-17 the id was dropped on the way in and the name was guessed
from the newest registry record, so the label was right only while registration and
deployment happened to agree.

**How do you know the model in the scoring path is the one being served?**
There is no copy to drift. The repository holds no model, manifest or threshold any
more: the scoring path resolves a registered bundle by the same rule the deploy script
applies (`select_serving_record`) and downloads the booster, the manifest and the
threshold from that one bundle into a cache keyed to it. Two files that belonged to
different trainings could sit side by side before, because they were files; now they
cannot be chosen separately. What is still stale is the fixture, which was built at
0.12 while a 0.12 bundle was current and has not been regenerated.

**Why is the fairness audit not a gate?**
Because it was decided, deliberately and in writing, that it should be a yellow light
rather than a red one: TPR and FPR parity per subgroup with confidence intervals,
surfaced to the reviewer, with a human deciding whether to deploy. Subgroup sizes are
small and drift is real, and a build that breaks on noise stops being run. The
honest cost of that choice is that an unfair model can be registered and deployed by
someone who does not read the audit.

**What would a reviewer say is missing from this layer?**
One thing stands between this layer and complete: the data reference says which state
was read but pins nothing re-readable, so a run can be traced to its data without
being reproduced from it. Beyond that, two defects of quality rather than of
capability: the feature contract is restated in five places, so a divergence arrives
as a missing value rather than an error, and the gate metrics are the one bundle file
the checksums do not cover. And there is no monitoring, so retraining is a person
remembering — a product decision, but a document that describes the loop as running
is not a substitute for it.
