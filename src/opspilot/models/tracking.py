"""MLflow setup: one local experiment, and every run tagged with the git commit."""

import subprocess

import mlflow

from opspilot.config import (
    MLFLOW_ARTIFACTS_DIR,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    REPO_ROOT,
)


def git_commit() -> str:
    """The commit the code ran from, marked -dirty when tracked files have changes."""
    try:
        commit = _git("rev-parse", "HEAD")
        dirty = _git("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{commit}-dirty" if dirty else commit


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def start_experiment() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT) is None:
        mlflow.create_experiment(MLFLOW_EXPERIMENT, artifact_location=MLFLOW_ARTIFACTS_DIR.as_uri())
    mlflow.set_experiment(MLFLOW_EXPERIMENT)


def log_period_metrics(results: dict[str, dict[str, float]]) -> None:
    """Log each period's metrics as <period>_<metric>, e.g. validation_pr_auc."""
    for period, metrics in results.items():
        mlflow.log_metrics({f"{period}_{name}": value for name, value in metrics.items()})
