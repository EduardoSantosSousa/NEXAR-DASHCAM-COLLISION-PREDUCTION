"""Lightweight MLflow helpers used by scripts and training utilities."""

from __future__ import annotations

import math
from contextlib import contextmanager
from numbers import Number
from pathlib import Path
from typing import Any, Iterator, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXPERIMENT_NAME = "nexar-collision-prediction"


def default_tracking_uri() -> str:
    """Return the local SQLite MLflow tracking URI."""
    db_path = (PROJECT_ROOT / "mlflow.db").resolve().as_posix()
    return f"sqlite:///{db_path}"


def import_mlflow():
    try:
        import mlflow  # type: ignore
    except ImportError:
        print(
            "MLflow is not installed. Run "
            "`./venv/Scripts/python.exe -m pip install -r requirements.txt` "
            "to enable experiment tracking."
        )
        return None
    return mlflow


@contextmanager
def mlflow_run(
    *,
    enabled: bool,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    run_name: str | None = None,
    tracking_uri: str | None = None,
    tags: Mapping[str, Any] | None = None,
) -> Iterator[Any | None]:
    """Start an MLflow run if tracking is enabled and MLflow is installed."""
    if not enabled:
        yield None
        return

    mlflow = import_mlflow()
    if mlflow is None:
        yield None
        return

    mlflow.set_tracking_uri(tracking_uri or default_tracking_uri())
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        if tags:
            mlflow.set_tags({key: str(value) for key, value in tags.items()})
        yield mlflow


def _clean_key(key: str) -> str:
    return (
        str(key)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


def _is_metric_value(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    if not isinstance(value, Number):
        return False
    return math.isfinite(float(value))


def log_params(mlflow: Any | None, params: Mapping[str, Any]) -> None:
    if mlflow is None:
        return

    clean_params = {}
    for key, value in params.items():
        if value is None:
            continue
        clean_params[_clean_key(key)] = str(value)

    if clean_params:
        mlflow.log_params(clean_params)


def log_metrics(
    mlflow: Any | None,
    metrics: Mapping[str, Any],
    *,
    prefix: str = "",
    step: int | None = None,
) -> None:
    if mlflow is None:
        return

    clean_metrics = {}
    for key, value in metrics.items():
        if _is_metric_value(value):
            clean_metrics[f"{prefix}{_clean_key(key)}"] = float(value)

    if clean_metrics:
        mlflow.log_metrics(clean_metrics, step=step)


def log_artifacts(
    mlflow: Any | None,
    paths: list[Path],
    *,
    artifact_path: str | None = None,
) -> None:
    if mlflow is None:
        return

    for path in paths:
        if path.exists():
            mlflow.log_artifact(str(path), artifact_path=artifact_path)
