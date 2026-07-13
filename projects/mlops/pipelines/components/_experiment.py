"""_experiment — best-effort logging to a companion Vertex AI Experiment run.

Why a *companion* run
---------------------
A pipeline submitted with ``job.submit(experiment=...)`` creates a
``system.PipelineRun`` context in the experiment.  That context only surfaces
(a) the pipeline's **input parameters** and (b) ``dsl.Metrics`` artifacts —
which the Experiments UI *always* files under the **Metrics** tab.  There is no
supported way to attach ``log_params()`` or time-series to a PipelineRun from
inside a component (the metadata context has the wrong schema, so an
``ExperimentRun`` lookup returns ``NotFound``).

Consequences that this module fixes:
  * tuned hyperparameters logged via ``dsl.Metrics`` get *bunched* into the
    Metrics tab instead of the Parameters tab;
  * per-Optuna-trial curves have nowhere to land, so the **Charts** tab is empty.

This module opens a dedicated ``ExperimentRun`` named ``"<job>-metrics"`` in the
same experiment.  On that run, ``log_params`` → Parameters UI, ``log_metrics`` →
Metrics UI, and ``log_time_series_metrics`` → Charts UI — each in its proper
place.  Every call is wrapped so telemetry can *never* abort a training run.

The module has no module-level ``google-cloud-aiplatform`` import and no KFP
decorators, so it is safe to import under the KFP executor (``_KFP_RUNTIME``).
"""

from contextlib import contextmanager

COMPANION_SUFFIX = "-metrics"


def companion_run_name(pipeline_job_name: str) -> str:
    """Distinct run name so it never collides with the PipelineRun context."""
    return f"{pipeline_job_name}{COMPANION_SUFFIX}"


@contextmanager
def companion_run(*, project_id, location, experiment, pipeline_job_name):
    """Yield the ``aiplatform`` module bound to an open companion run, or ``None``.

    Resumes the run if a prior step already created it; otherwise creates it.
    Yields ``None`` (a harmless no-op for callers) if anything goes wrong or the
    required identifiers are missing.
    """
    if not (project_id and experiment and pipeline_job_name):
        yield None
        return
    try:
        from google.cloud import aiplatform
    except Exception as exc:  # noqa: BLE001 - telemetry must never break training
        print(f"  [warn] aiplatform unavailable; experiment logging skipped: {exc}")
        yield None
        return

    name = companion_run_name(pipeline_job_name)
    opened = False
    try:
        aiplatform.init(project=project_id, location=location, experiment=experiment)
        try:
            aiplatform.start_run(name, resume=True)  # resume if a prior step made it
        except Exception:
            aiplatform.start_run(name)  # otherwise create it (resume=False default)
        opened = True
        yield aiplatform
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] companion experiment run unavailable; logging skipped: {exc}")
        yield None
    finally:
        if opened:
            try:
                aiplatform.end_run()
            except Exception:  # noqa: BLE001
                pass


def safe_log_params(ap, params: dict) -> None:
    """Best-effort ``log_params`` (Parameters UI). No-op if ``ap`` is None."""
    if ap is None:
        return
    try:
        ap.log_params({k: v for k, v in params.items() if v is not None})
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] log_params skipped: {exc}")


def safe_log_metrics(ap, metrics: dict) -> None:
    """Best-effort ``log_metrics`` (Metrics UI). No-op if ``ap`` is None."""
    if ap is None:
        return
    try:
        ap.log_metrics({k: float(v) for k, v in metrics.items() if v is not None})
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] log_metrics skipped: {exc}")
