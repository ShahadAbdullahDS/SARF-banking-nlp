"""Reference TF-IDF + LinearSVC baseline for the Fahimt project.

This baseline must train on the approved MSA train split only. Model
selection uses MSA validation only. The frozen Saudi split is reserved
for the planned final cross-dialect evaluation.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

try:
    from .metrics import save_evaluation_artifacts
except ImportError:
    from metrics import save_evaluation_artifacts


REQUIRED_COLUMNS = {"text", "label"}
MODEL_NAME = "tfidf_linear_svc"


def set_seed(seed: int) -> None:
    """Set seeds for reproducible data handling and metadata."""
    random.seed(seed)
    np.random.seed(seed)


def load_processed_split(csv_path: str | Path) -> pd.DataFrame:
    """Load an approved processed split without silently modifying it."""
    csv_path = Path(csv_path)
    data = pd.read_csv(csv_path)

    missing_columns = REQUIRED_COLUMNS - set(data.columns)
    if missing_columns:
        raise ValueError(f"{csv_path.name} is missing columns: {sorted(missing_columns)}")
    if data.empty:
        raise ValueError(f"{csv_path.name} contains zero rows.")
    if data["label"].isna().any():
        raise ValueError(f"{csv_path.name} contains missing labels.")
    if data["text"].isna().any() or data["text"].astype(str).str.strip().eq("").any():
        raise ValueError(f"{csv_path.name} contains missing or blank text.")

    return data.copy()


def build_svm_pipeline() -> Pipeline:
    """Create the fixed A1 baseline pipeline; do not tune on Saudi test."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=100_000,
                    sublinear_tf=True,
                    lowercase=False,
                ),
            ),
            ("classifier", LinearSVC(class_weight="balanced")),
        ]
    )


def run_svm_baseline(
    train_csv: str | Path,
    evaluation_csv: str | Path,
    output_dir: str | Path,
    run_id: str,
    evaluation_split: str,
    seed: int = 42,
) -> dict[str, Any]:
    """Train the A1 baseline and save all required evaluation artifacts."""
    set_seed(seed)
    train_data = load_processed_split(train_csv)
    evaluation_data = load_processed_split(evaluation_csv)

    label_order = sorted(train_data["label"].unique().tolist(), key=str)
    pipeline = build_svm_pipeline()
    pipeline.fit(train_data["text"], train_data["label"])
    predictions = pipeline.predict(evaluation_data["text"])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_metadata = {
        "run_id": run_id,
        "model": MODEL_NAME,
        "seed": seed,
        "train_split": Path(train_csv).name,
        "evaluation_split": evaluation_split,
        "evaluation_file": Path(evaluation_csv).name,
        "training_examples": int(len(train_data)),
        "evaluation_examples": int(len(evaluation_data)),
        "protocol_note": (
            "MSA validation is for model selection; the Saudi frozen test "
            "must not be used for tuning or feature decisions."
        ),
    }

    artifact_paths = save_evaluation_artifacts(
        output_dir=output_dir,
        y_true=evaluation_data["label"].tolist(),
        y_pred=predictions.tolist(),
        label_order=label_order,
        run_metadata=run_metadata,
        sample_ids=evaluation_data.index.tolist(),
        texts=evaluation_data["text"].tolist(),
    )

    model_path = output_dir / "model.joblib"
    config_path = output_dir / "run_config.json"
    joblib.dump(pipeline, model_path)
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                **run_metadata,
                "label_order": label_order,
                "tfidf": {
                    "analyzer": "word",
                    "ngram_range": [1, 2],
                    "min_df": 1,
                    "max_features": 100_000,
                    "sublinear_tf": True,
                    "lowercase": False,
                },
                "linear_svc": {"class_weight": "balanced"},
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    artifact_paths.update({"model": model_path, "config": config_path})
    return artifact_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Fahimt A1 SVM baseline.")
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--evaluation_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument(
        "--evaluation_split",
        required=True,
        choices=["msa_val", "msa_test", "saudi_frozen_test"],
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    paths = run_svm_baseline(
        train_csv=args.train_csv,
        evaluation_csv=args.evaluation_csv,
        output_dir=args.output_dir,
        run_id=args.run_id,
        evaluation_split=args.evaluation_split,
        seed=args.seed,
    )
    print("Saved artifacts:")
    for artifact_name, artifact_path in paths.items():
        print(f"- {artifact_name}: {artifact_path}")
