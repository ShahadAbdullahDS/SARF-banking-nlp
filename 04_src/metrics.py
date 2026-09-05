"""Shared evaluation utilities for the Fahimt Banking NLP project.

Primary metric: Macro-F1.
Secondary metrics: weighted F1 and accuracy.

Every experiment must use the same fixed label order loaded from the
MSA training split, and must save metrics, per-intent metrics, and
predictions using these utilities.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

PRIMARY_METRIC = "macro_f1"
SECONDARY_METRICS = ("weighted_f1", "accuracy")


def _as_list(values: Iterable[Any]) -> list[Any]:
    """Convert an iterable to a list and reject empty inputs."""
    values = list(values)
    if not values:
        raise ValueError("Evaluation received zero examples.")
    if any(pd.isna(value) for value in values):
        raise ValueError("Evaluation labels or predictions contain missing values.")
    return values


def validate_evaluation_inputs(
    y_true: Iterable[Any],
    y_pred: Iterable[Any],
    label_order: Sequence[Any],
) -> tuple[list[Any], list[Any], list[Any]]:
    """Validate aligned labels, predictions, and the fixed label order."""
    y_true = _as_list(y_true)
    y_pred = _as_list(y_pred)
    label_order = _as_list(label_order)

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true has {len(y_true)} rows but y_pred has {len(y_pred)} rows."
        )

    unknown_true = sorted(set(y_true) - set(label_order), key=str)
    unknown_pred = sorted(set(y_pred) - set(label_order), key=str)
    if unknown_true:
        raise ValueError(f"Ground-truth labels outside label_order: {unknown_true}")
    if unknown_pred:
        raise ValueError(f"Predicted labels outside label_order: {unknown_pred}")

    return y_true, y_pred, label_order


def compute_core_metrics(
    y_true: Iterable[Any],
    y_pred: Iterable[Any],
    label_order: Sequence[Any],
) -> dict[str, float | int]:
    """Compute the project's predefined core metrics with a fixed label order."""
    y_true, y_pred, label_order = validate_evaluation_inputs(y_true, y_pred, label_order)

    return {
        "n_examples": len(y_true),
        "n_labels_in_protocol": len(label_order),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=label_order, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=label_order, average="weighted", zero_division=0)
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def build_per_intent_metrics(
    y_true: Iterable[Any],
    y_pred: Iterable[Any],
    label_order: Sequence[Any],
) -> pd.DataFrame:
    """Return precision, recall, F1, and support for every protocol label."""
    y_true, y_pred, label_order = validate_evaluation_inputs(y_true, y_pred, label_order)

    report = classification_report(
        y_true,
        y_pred,
        labels=label_order,
        target_names=[str(label) for label in label_order],
        output_dict=True,
        zero_division=0,
    )

    rows = []
    for label in label_order:
        values = report[str(label)]
        rows.append(
            {
                "label": label,
                "precision": float(values["precision"]),
                "recall": float(values["recall"]),
                "f1": float(values["f1-score"]),
                "support": int(values["support"]),
            }
        )

    return pd.DataFrame(rows)


def save_evaluation_artifacts(
    output_dir: str | Path,
    y_true: Iterable[Any],
    y_pred: Iterable[Any],
    label_order: Sequence[Any],
    run_metadata: dict[str, Any],
    sample_ids: Iterable[Any] | None = None,
    texts: Iterable[str] | None = None,
) -> dict[str, Path]:
    """Save reproducible metrics, per-intent metrics, predictions, and run metadata."""
    y_true, y_pred, label_order = validate_evaluation_inputs(y_true, y_pred, label_order)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = compute_core_metrics(y_true, y_pred, label_order)
    metrics.update(run_metadata)

    predictions = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    if sample_ids is not None:
        sample_ids = list(sample_ids)
        if len(sample_ids) != len(predictions):
            raise ValueError("sample_ids length does not match predictions length.")
        predictions.insert(0, "sample_id", sample_ids)
    if texts is not None:
        texts = list(texts)
        if len(texts) != len(predictions):
            raise ValueError("texts length does not match predictions length.")
        predictions.insert(len(predictions.columns), "text", texts)

    paths = {
        "metrics": output_dir / "metrics.json",
        "per_intent": output_dir / "per_intent_metrics.csv",
        "predictions": output_dir / "predictions.csv",
    }

    with paths["metrics"].open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2, default=str)
    build_per_intent_metrics(y_true, y_pred, label_order).to_csv(
        paths["per_intent"], index=False, encoding="utf-8"
    )
    predictions.to_csv(paths["predictions"], index=False, encoding="utf-8")

    return paths
