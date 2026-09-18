"""One-time final held-out Saudi evaluation for SARF baselines and AraBERT.

Modes
-----
``preflight`` validates all frozen model artifacts and MSA-validation evidence.
It does not open the Saudi test.

``official`` opens the held-out Saudi test once, evaluates the frozen SVM,
TextCNN, and fifteen AraBERT checkpoints, writes per-model evidence, creates
combined summaries, and permanently locks the final evaluation against reruns.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as functional
from arabert.preprocess import ArabertPreprocessor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

OWNER = "A"
CONDITIONS = ("E0", "E1", "E2", "E3", "EB")
SEEDS = (42, 123, 2026)
NUM_LABELS = 77
ARABERT_BASE_CHECKPOINT = "aubmindlab/bert-base-arabertv2"
TEXTCNN_MAX_LENGTH = 64
TEXTCNN_EMBEDDING_DIM = 128
TEXTCNN_FILTER_SIZES = (3, 4, 5)
TEXTCNN_FILTERS_PER_SIZE = 100
TEXTCNN_DROPOUT = 0.5


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def arabert_config(self) -> Path:
        return self.root / "02_configs" / "arabert_config.json"

    @property
    def arabert_mapping(self) -> Path:
        return self.root / "02_configs" / "arabert_label_mapping.json"

    @property
    def arabert_runs(self) -> Path:
        return self.root / "05_runs" / "arabert"

    @property
    def arabert_models(self) -> Path:
        return self.root / "06_models_and_checkpoints" / "arabert"

    @property
    def baseline_runs(self) -> Path:
        return self.root / "05_runs" / "baselines"

    @property
    def baseline_models(self) -> Path:
        return self.root / "06_models_and_checkpoints" / "baselines"

    @property
    def result_root(self) -> Path:
        return self.root / "07_results" / "final_saudi_evaluation"

    @property
    def preflight_path(self) -> Path:
        return self.result_root / "final_saudi_preflight.json"

    @property
    def lock_path(self) -> Path:
        return self.result_root / "final_saudi_evaluation_in_progress.json"

    @property
    def manifest_path(self) -> Path:
        return self.result_root / "final_saudi_evaluation_manifest.json"

    def arabert_run_dir(self, condition: str, seed: int) -> Path:
        return self.arabert_runs / condition / f"seed_{seed}"

    def arabert_checkpoint(self, condition: str, seed: int) -> Path:
        return self.arabert_models / condition / f"seed_{seed}" / "best_checkpoint"

    def arabert_saudi_output(self, condition: str, seed: int) -> Path:
        return self.arabert_run_dir(condition, seed) / "saudi_eval"

    @property
    def svm_model(self) -> Path:
        return self.baseline_models / "SVM" / "svm_tfidf.joblib"

    @property
    def textcnn_model(self) -> Path:
        return self.baseline_models / "textCnn" / "textcnn_best.pt"

    @property
    def textcnn_vocabulary(self) -> Path:
        return self.baseline_models / "textCnn" / "vocabulary.json"

    @property
    def textcnn_mapping(self) -> Path:
        return self.baseline_models / "textCnn" / "label_mapping.json"

    @property
    def svm_output(self) -> Path:
        return self.baseline_runs / "SVM" / "saudi_eval"

    @property
    def textcnn_output(self) -> Path:
        return self.baseline_runs / "textCnn" / "saudi_eval"

    @property
    def held_out_test(self) -> Path:
        """The frozen Saudi file is reachable only from official-evaluation code."""
        return self.root / "02_processed_data" / "saudi_test_frozen_v1.csv"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_gpu() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for final TextCNN and AraBERT evaluation.")


def environment_record() -> dict[str, str]:
    return {
        "python": sys.version.replace("\n", " "),
        "torch": torch.__version__,
        "transformers": version("transformers"),
        "arabert": version("arabert"),
        "scikit_learn": version("scikit-learn"),
        "gpu": torch.cuda.get_device_name(0),
    }


def find_metric_payload(directory: Path, model_id: str) -> dict[str, float]:
    """Find the frozen MSA-validation metrics that establish the pre-test reference."""
    candidates = [
        directory / "evaluation_summary.json",
        directory / "validation_metrics.json",
        directory / "metrics.json",
        directory / "summary.json",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        payload = read_json(path)
        metrics = payload.get("metrics", payload)
        keys = {"macro_f1", "weighted_f1", "accuracy"}
        if isinstance(metrics, dict) and keys.issubset(metrics):
            return {name: float(metrics[name]) for name in keys}
    raise AssertionError(
        f"MSA-validation metric evidence not found for {model_id}. Checked:\n" + "\n".join(map(str, candidates))
    )


def load_arabert_mapping(paths: Paths) -> tuple[dict[str, int], dict[int, str]]:
    assert paths.arabert_mapping.is_file(), f"Missing AraBERT label mapping: {paths.arabert_mapping}"
    payload = read_json(paths.arabert_mapping)
    assert payload["owner"] == OWNER
    assert payload["num_labels"] == NUM_LABELS
    label_to_id = {str(label): int(label_id) for label, label_id in payload["label_to_id"].items()}
    id_to_label = {int(label_id): str(label) for label_id, label in payload["id_to_label"].items()}
    assert set(label_to_id.values()) == set(range(NUM_LABELS))
    assert set(id_to_label) == set(range(NUM_LABELS))
    assert all(id_to_label[label_id] == label for label, label_id in label_to_id.items())
    return label_to_id, id_to_label


def validate_arabert(paths: Paths, label_to_id: dict[str, int]) -> list[dict[str, Any]]:
    assert paths.arabert_config.is_file(), f"Missing final AraBERT config: {paths.arabert_config}"
    config = read_json(paths.arabert_config)
    assert config["owner"] == OWNER
    assert config["base_checkpoint"] == ARABERT_BASE_CHECKPOINT
    assert config["official_conditions"] == list(CONDITIONS)
    assert config["official_seeds"] == list(SEEDS)
    assert config["fine_tuning"]["learning_rate"] == 2e-5
    evidence: list[dict[str, Any]] = []
    missing: list[str] = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            checkpoint = paths.arabert_checkpoint(condition, seed)
            run_dir = paths.arabert_run_dir(condition, seed)
            original_summary = run_dir / "evaluation_summary.json"
            required = [
                checkpoint / "config.json",
                checkpoint / "model.safetensors",
                checkpoint / "tokenizer.json",
                checkpoint / "tokenizer_config.json",
                checkpoint / "vocab.txt",
                original_summary,
            ]
            absent = [str(path) for path in required if not path.is_file()]
            if absent:
                missing.extend(absent)
                continue
            summary = read_json(original_summary)
            assert summary["owner"] == OWNER
            assert summary["run_id"] == f"{condition}_seed_{seed}"
            assert summary["condition"] == condition
            assert summary["seed"] == seed
            assert summary["num_labels"] == NUM_LABELS
            assert summary["checkpoint_selection_metric"] == "macro_f1 on MSA validation"
            assert summary["checks"]["checkpoint_reload_passed"] is True
            assert summary["saudi_test_accessed"] is False
            evidence.append(
                {
                    "model_id": f"AraBERT_{condition}_seed_{seed}",
                    "model_family": "AraBERTv2-base",
                    "condition": condition,
                    "seed": seed,
                    "checkpoint": str(checkpoint),
                    "checkpoint_model_sha256": sha256(checkpoint / "model.safetensors"),
                    "checkpoint_config_sha256": sha256(checkpoint / "config.json"),
                    "msa_validation_metrics": summary["metrics"],
                }
            )
    assert not missing, "Missing AraBERT final artifacts:\n" + "\n".join(missing)
    assert len(evidence) == len(CONDITIONS) * len(SEEDS)
    return evidence


def validate_svm(paths: Paths, labels: set[str]) -> dict[str, Any]:
    assert paths.svm_model.is_file(), f"Missing SVM pipeline: {paths.svm_model}"
    pipeline = joblib.load(paths.svm_model)
    assert callable(getattr(pipeline, "predict", None))
    assert list(pipeline.named_steps) == ["tfidf", "classifier"]
    assert pipeline.named_steps["tfidf"].__class__.__name__ == "TfidfVectorizer"
    assert pipeline.named_steps["classifier"].__class__.__name__ == "LinearSVC"
    assert len(pipeline.classes_) == NUM_LABELS
    assert set(map(str, pipeline.classes_)) == labels
    metrics = find_metric_payload(paths.baseline_runs / "svm_msa_validation_v1", "SVM")
    return {
        "model_id": "TFIDF_LinearSVC",
        "model_family": "TF-IDF + LinearSVC",
        "condition": "baseline",
        "seed": None,
        "checkpoint": str(paths.svm_model),
        "checkpoint_sha256": sha256(paths.svm_model),
        "msa_validation_metrics": metrics,
    }


class TextCNN(nn.Module):
    """Exact recovered TextCNN architecture from the frozen checkpoint tensors."""

    def __init__(self, vocabulary_size: int, pad_id: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, TEXTCNN_EMBEDDING_DIM, padding_idx=pad_id)
        self.convolutions = nn.ModuleList(
            [nn.Conv1d(TEXTCNN_EMBEDDING_DIM, TEXTCNN_FILTERS_PER_SIZE, kernel_size) for kernel_size in TEXTCNN_FILTER_SIZES]
        )
        self.dropout = nn.Dropout(TEXTCNN_DROPOUT)
        self.classifier = nn.Linear(TEXTCNN_FILTERS_PER_SIZE * len(TEXTCNN_FILTER_SIZES), NUM_LABELS)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids).transpose(1, 2)
        pooled = []
        for layer in self.convolutions:
            activation = functional.relu(layer(embedded))
            pooled.append(functional.max_pool1d(activation, kernel_size=activation.shape[2]).squeeze(2))
        return self.classifier(self.dropout(torch.cat(pooled, dim=1)))


def validate_textcnn(paths: Paths, label_to_id: dict[str, int]) -> dict[str, Any]:
    required = [paths.textcnn_model, paths.textcnn_vocabulary, paths.textcnn_mapping]
    absent = [str(path) for path in required if not path.is_file()]
    assert not absent, "Missing TextCNN artifacts:\n" + "\n".join(absent)
    checkpoint = torch.load(paths.textcnn_model, map_location="cpu", weights_only=True)
    assert isinstance(checkpoint, dict)
    assert set(checkpoint) == {"model_state_dict", "best_epoch", "best_macro_f1", "config"}
    config = checkpoint["config"]
    assert config["model_family"] == "TextCNN"
    assert config["num_labels"] == NUM_LABELS
    assert config["vocab_size"] == 5755
    assert config["tokenizer"] == "Unicode word regex"
    assert config["max_length"] == TEXTCNN_MAX_LENGTH
    assert config["embedding_dimension"] == TEXTCNN_EMBEDDING_DIM
    assert tuple(config["filter_sizes"]) == TEXTCNN_FILTER_SIZES
    assert config["filters_per_size"] == TEXTCNN_FILTERS_PER_SIZE
    assert config["dropout"] == TEXTCNN_DROPOUT
    assert config["saudi_test_accessed"] is False
    vocabulary = read_json(paths.textcnn_vocabulary)
    textcnn_mapping = read_json(paths.textcnn_mapping)
    saved_mapping = {str(label): int(label_id) for label, label_id in textcnn_mapping["label_to_id"].items()}
    saved_id_to_label = {int(label_id): str(label) for label_id, label in textcnn_mapping["id_to_label"].items()}
    assert vocabulary["<PAD>"] == 0
    assert vocabulary["<UNK>"] == 1
    assert len(vocabulary) == config["vocab_size"]
    assert set(saved_mapping) == set(label_to_id), "TextCNN labels differ from the shared 77-label label set."
    assert set(saved_id_to_label) == set(range(NUM_LABELS))
    assert all(saved_id_to_label[label_id] == label for label, label_id in saved_mapping.items())
    state_dict = checkpoint["model_state_dict"]
    expected_shapes = {
        "embedding.weight": (5755, 128),
        "convolutions.0.weight": (100, 128, 3),
        "convolutions.1.weight": (100, 128, 4),
        "convolutions.2.weight": (100, 128, 5),
        "classifier.weight": (77, 300),
        "classifier.bias": (77,),
    }
    for name, shape in expected_shapes.items():
        assert tuple(state_dict[name].shape) == shape, f"Unexpected TextCNN shape: {name}"
    metrics = find_metric_payload(paths.baseline_runs / "textcnn_msa_validation_v1", "TextCNN")
    return {
        "model_id": "TextCNN",
        "model_family": "TextCNN",
        "condition": "baseline",
        "seed": int(config["seed"]),
        "checkpoint": str(paths.textcnn_model),
        "checkpoint_sha256": sha256(paths.textcnn_model),
        "msa_validation_metrics": metrics,
        "textcnn_config": config,
        "textcnn_id_to_label": saved_id_to_label,
    }


def validate_no_prior_evaluation(paths: Paths) -> None:
    assert not paths.manifest_path.exists(), "Final Saudi evaluation is already complete and must not be rerun."
    assert not paths.lock_path.exists(), "Final Saudi evaluation lock exists; do not start a second run."
    preexisting: list[str] = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            output = paths.arabert_saudi_output(condition, seed)
            if output.exists():
                preexisting.append(str(output))
    for output in (paths.svm_output, paths.textcnn_output):
        if output.exists():
            preexisting.append(str(output))
    assert not preexisting, "Saudi evaluation output already exists; do not overwrite:\n" + "\n".join(preexisting)


def preflight(project_root: str | Path) -> dict[str, Any]:
    paths = Paths(Path(project_root))
    assert paths.root.is_dir(), f"Project root not found: {paths.root}"
    require_gpu()
    validate_no_prior_evaluation(paths)
    label_to_id, id_to_label = load_arabert_mapping(paths)
    arabert = validate_arabert(paths, label_to_id)
    svm = validate_svm(paths, set(label_to_id))
    textcnn = validate_textcnn(paths, label_to_id)
    payload = {
        "owner": OWNER,
        "status": "preflight_passed_no_saudi_test_access",
        "evaluation_scope": {
            "svm_models": 1,
            "textcnn_models": 1,
            "arabert_models": len(arabert),
            "total_models": len(arabert) + 2,
        },
        "num_labels": NUM_LABELS,
        "arabert_config": str(paths.arabert_config),
        "arabert_mapping": str(paths.arabert_mapping),
        "arabert_checkpoints": arabert,
        "svm": svm,
        "textcnn": textcnn,
        "environment": environment_record(),
        "saudi_test_accessed": False,
    }
    paths.result_root.mkdir(parents=True, exist_ok=True)
    write_json(paths.preflight_path, payload)
    return payload


def read_held_out_test(paths: Paths, label_to_id: dict[str, int]) -> tuple[pd.DataFrame, dict[str, Any]]:
    test_path = paths.held_out_test
    assert test_path.is_file(), f"Missing Saudi test: {test_path}"
    frame = pd.read_csv(test_path)
    assert {"label", "text"}.issubset(frame.columns), "Saudi test must have label and text columns."
    assert frame["label"].notna().all() and frame["text"].notna().all()
    assert frame["text"].astype(str).str.strip().ne("").all(), "Saudi test has blank text."
    frame = frame.copy()
    frame["label"] = frame["label"].astype(str)
    assert not (set(frame["label"]) - set(label_to_id)), "Saudi test has unmapped labels."
    frame.insert(0, "row_id", np.arange(len(frame), dtype=int))
    frame["label_id"] = frame["label"].map(label_to_id).astype(int)
    return frame, {
        "relative_path": "02_processed_data/saudi_test_frozen_v1.csv",
        "sha256": sha256(test_path),
        "rows": int(len(frame)),
        "opened_at_utc": utc_now(),
    }


def compute_metrics(true_ids: np.ndarray, predicted_ids: np.ndarray) -> dict[str, float]:
    labels = np.arange(NUM_LABELS)
    return {
        "macro_f1": float(f1_score(true_ids, predicted_ids, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(true_ids, predicted_ids, labels=labels, average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(true_ids, predicted_ids)),
    }


def save_model_outputs(
    destination: Path,
    model_metadata: dict[str, Any],
    test_frame: pd.DataFrame,
    predicted_ids: np.ndarray,
    id_to_label: dict[int, str],
    test_metadata: dict[str, Any],
    preprocessing: dict[str, Any],
) -> dict[str, Any]:
    assert not destination.exists(), f"Refusing to overwrite final output: {destination}"
    destination.mkdir(parents=True, exist_ok=False)
    true_ids = test_frame["label_id"].to_numpy()
    assert predicted_ids.shape == true_ids.shape
    assert np.unique(predicted_ids).size > 1, f"{model_metadata['model_id']} predicted one class only."
    metrics = compute_metrics(true_ids, predicted_ids)
    labels = [id_to_label[index] for index in range(NUM_LABELS)]
    predicted_labels = [id_to_label[int(predicted_id)] for predicted_id in predicted_ids]
    prediction_frame = pd.DataFrame(
        {
            "row_id": test_frame["row_id"].to_numpy(),
            "text": test_frame["text"].to_numpy(),
            "true_label": test_frame["label"].to_numpy(),
            "true_label_id": true_ids,
            "predicted_label": predicted_labels,
            "predicted_label_id": predicted_ids,
            "is_correct": test_frame["label"].to_numpy() == np.array(predicted_labels),
        }
    )
    prediction_frame.to_csv(destination / "saudi_predictions.csv", index=False)
    report = classification_report(true_ids, predicted_ids, labels=np.arange(NUM_LABELS), target_names=labels, output_dict=True, zero_division=0)
    pd.DataFrame(
        [
            {
                "label": label,
                "label_id": label_id,
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1_score": report[label]["f1-score"],
                "support": report[label]["support"],
            }
            for label_id, label in enumerate(labels)
        ]
    ).to_csv(destination / "saudi_per_class_metrics.csv", index=False)
    matrix = confusion_matrix(true_ids, predicted_ids, labels=np.arange(NUM_LABELS))
    matrix_frame = pd.DataFrame(matrix, index=labels, columns=labels)
    matrix_frame.index.name = "true_label"
    matrix_frame.columns.name = "predicted_label"
    matrix_frame.to_csv(destination / "saudi_confusion_matrix.csv")
    msa = model_metadata["msa_validation_metrics"]
    payload = {
        **model_metadata,
        "owner": OWNER,
        "evaluation_type": "one_time_final_held_out_saudi_evaluation",
        "test_split": "saudi_test",
        "test_rows": int(len(test_frame)),
        "num_labels": NUM_LABELS,
        "primary_metric": "macro_f1",
        "metrics": metrics,
        "msa_to_saudi_metric_change": {
            name: float(metrics[name] - float(msa[name]))
            for name in ("macro_f1", "weighted_f1", "accuracy")
        },
        "preprocessing": preprocessing,
        "artifacts": {
            "predictions": "saudi_predictions.csv",
            "per_class_metrics": "saudi_per_class_metrics.csv",
            "confusion_matrix": "saudi_confusion_matrix.csv",
        },
        "checks": {
            "all_77_labels_used_for_metric": True,
            "predictions_not_single_label": True,
            "frozen_checkpoint_or_pipeline": True,
            "shared_label_mapping_verified": True,
        },
        "test_access_metadata": test_metadata,
        "saudi_test_accessed": True,
        "status": "completed",
    }
    write_json(destination / "saudi_eval_metrics.json", payload)
    return payload


def predict_svm(paths: Paths, text: list[str], label_to_id: dict[str, int]) -> np.ndarray:
    pipeline = joblib.load(paths.svm_model)
    predicted_labels = pipeline.predict(text)
    assert set(map(str, pipeline.classes_)) == set(label_to_id)
    return np.array([label_to_id[str(label)] for label in predicted_labels], dtype=int)


def unicode_word_tokenize(text: str) -> list[str]:
    """The recorded TextCNN tokenizer: Unicode word regex, preserving Arabic words and digits."""
    return re.findall(r"\w+", str(text), flags=re.UNICODE)


def encode_textcnn(text: list[str], vocabulary: dict[str, int]) -> torch.Tensor:
    pad_id = int(vocabulary["<PAD>"])
    unknown_id = int(vocabulary["<UNK>"])
    encoded: list[list[int]] = []
    for value in text:
        token_ids = [int(vocabulary.get(token, unknown_id)) for token in unicode_word_tokenize(value)]
        token_ids = token_ids[:TEXTCNN_MAX_LENGTH]
        encoded.append(token_ids + [pad_id] * (TEXTCNN_MAX_LENGTH - len(token_ids)))
    return torch.tensor(encoded, dtype=torch.long)


def predict_textcnn(
    paths: Paths,
    text: list[str],
    label_to_id: dict[str, int],
    textcnn_id_to_label: dict[int, str],
) -> np.ndarray:
    checkpoint = torch.load(paths.textcnn_model, map_location="cpu", weights_only=True)
    vocabulary = {str(token): int(token_id) for token, token_id in read_json(paths.textcnn_vocabulary).items()}
    model = TextCNN(len(vocabulary), int(vocabulary["<PAD>"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to("cuda").eval()
    tokens = encode_textcnn(text, vocabulary)
    loader = DataLoader(TensorDataset(tokens), batch_size=64, shuffle=False)
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for (batch,) in loader:
            probabilities.append(model(batch.to("cuda")).cpu().numpy())
    logits = np.concatenate(probabilities, axis=0)
    assert logits.shape == (len(text), NUM_LABELS)
    del model
    torch.cuda.empty_cache()
    raw_ids = np.argmax(logits, axis=1)
    return np.array([label_to_id[textcnn_id_to_label[int(raw_id)]] for raw_id in raw_ids], dtype=int)


def prepare_arabert(text: list[str], tokenizer: Any) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    preprocessor = ArabertPreprocessor(model_name=ARABERT_BASE_CHECKPOINT)
    prepared = [preprocessor.preprocess(str(value)) for value in text]
    assert all(value.strip() for value in prepared)
    raw_ids = tokenizer(prepared, add_special_tokens=True, truncation=False)["input_ids"]
    lengths = np.array([len(ids) for ids in raw_ids])
    tokens = tokenizer(prepared, padding=True, truncation=True, max_length=128, return_tensors="pt")
    return tokens, {
        "method": "ArabertPreprocessor",
        "model_name": ARABERT_BASE_CHECKPOINT,
        "max_length": 128,
        "test_truncation_rate": float(np.mean(lengths > 128)),
        "test_max_untruncated_length": int(lengths.max()),
    }


def predict_arabert(checkpoint: Path, tokenized: dict[str, torch.Tensor], label_to_id: dict[str, int]) -> np.ndarray:
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to("cuda").eval()
    assert model.config.num_labels == NUM_LABELS
    assert {str(label): int(label_id) for label, label_id in model.config.label2id.items()} == label_to_id
    loader = DataLoader(TensorDataset(tokenized["input_ids"], tokenized["attention_mask"]), batch_size=32, shuffle=False)
    batches: list[np.ndarray] = []
    with torch.no_grad():
        for input_ids, attention_mask in loader:
            batches.append(model(input_ids=input_ids.to("cuda"), attention_mask=attention_mask.to("cuda")).logits.cpu().numpy())
    logits = np.concatenate(batches, axis=0)
    assert logits.shape[1] == NUM_LABELS
    del model
    torch.cuda.empty_cache()
    return np.argmax(logits, axis=1)


def make_summary(rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_frame = pd.DataFrame(rows).sort_values(["model_family", "condition", "seed"], na_position="first").reset_index(drop=True)
    aggregate_rows: list[dict[str, Any]] = []
    for model_family, condition in [("TF-IDF + LinearSVC", "baseline"), ("TextCNN", "baseline")]:
        subset = seed_frame[(seed_frame["model_family"] == model_family) & (seed_frame["condition"] == condition)]
        assert len(subset) == 1
        row = subset.iloc[0]
        aggregate_rows.append(
            {
                "model": model_family,
                "model_family": model_family,
                "condition": condition,
                "seed_count": 1,
                "macro_f1_mean": row["saudi_macro_f1"],
                "macro_f1_std": 0.0,
                "weighted_f1_mean": row["saudi_weighted_f1"],
                "weighted_f1_std": 0.0,
                "accuracy_mean": row["saudi_accuracy"],
                "accuracy_std": 0.0,
                "msa_validation_macro_f1_mean": row["msa_validation_macro_f1"],
                "macro_f1_change_msa_to_saudi": row["saudi_macro_f1"] - row["msa_validation_macro_f1"],
            }
        )
    for condition in CONDITIONS:
        subset = seed_frame[(seed_frame["model_family"] == "AraBERTv2-base") & (seed_frame["condition"] == condition)]
        assert len(subset) == len(SEEDS)
        aggregate_rows.append(
            {
                "model": f"AraBERT {condition}",
                "model_family": "AraBERTv2-base",
                "condition": condition,
                "seed_count": len(subset),
                "macro_f1_mean": subset["saudi_macro_f1"].mean(),
                "macro_f1_std": subset["saudi_macro_f1"].std(ddof=1),
                "weighted_f1_mean": subset["saudi_weighted_f1"].mean(),
                "weighted_f1_std": subset["saudi_weighted_f1"].std(ddof=1),
                "accuracy_mean": subset["saudi_accuracy"].mean(),
                "accuracy_std": subset["saudi_accuracy"].std(ddof=1),
                "msa_validation_macro_f1_mean": subset["msa_validation_macro_f1"].mean(),
                "macro_f1_change_msa_to_saudi": subset["saudi_macro_f1"].mean() - subset["msa_validation_macro_f1"].mean(),
            }
        )
    aggregate = pd.DataFrame(aggregate_rows).sort_values("macro_f1_mean", ascending=False).reset_index(drop=True)
    return seed_frame, aggregate


def official(project_root: str | Path) -> dict[str, Any]:
    paths = Paths(Path(project_root))
    require_gpu()
    validate_no_prior_evaluation(paths)
    label_to_id, id_to_label = load_arabert_mapping(paths)
    arabert_models = validate_arabert(paths, label_to_id)
    svm_model = validate_svm(paths, set(label_to_id))
    textcnn_model = validate_textcnn(paths, label_to_id)
    paths.result_root.mkdir(parents=True, exist_ok=True)
    write_json(paths.lock_path, {
        "owner": OWNER,
        "status": "in_progress",
        "started_at_utc": utc_now(),
        "evaluation_scope": {"svm": 1, "textcnn": 1, "arabert": 15},
        "saudi_test_accessed": False,
    })
    test_opened = False
    try:
        assert paths.held_out_test.is_file(), f"Missing Saudi test: {paths.held_out_test}"
        test_opened = True
        lock = read_json(paths.lock_path)
        lock.update({
            "saudi_test_accessed": True,
            "test_file": "02_processed_data/saudi_test_frozen_v1.csv",
            "test_opened_at_utc": utc_now(),
        })
        write_json(paths.lock_path, lock)
        test_frame, test_metadata = read_held_out_test(paths, label_to_id)
        rows: list[dict[str, Any]] = []
        svm_predictions = predict_svm(paths, test_frame["text"].tolist(), label_to_id)
        svm_output = save_model_outputs(paths.svm_output, svm_model, test_frame, svm_predictions, id_to_label, test_metadata, {"method": "frozen TfidfVectorizer inside sklearn Pipeline"})
        rows.append({
            "model_id": svm_model["model_id"], "model_family": svm_model["model_family"], "condition": "baseline", "seed": None,
            "msa_validation_macro_f1": svm_model["msa_validation_metrics"]["macro_f1"],
            "saudi_macro_f1": svm_output["metrics"]["macro_f1"], "saudi_weighted_f1": svm_output["metrics"]["weighted_f1"], "saudi_accuracy": svm_output["metrics"]["accuracy"],
        })
        textcnn_predictions = predict_textcnn(
            paths,
            test_frame["text"].tolist(),
            label_to_id,
            textcnn_model["textcnn_id_to_label"],
        )
        textcnn_output = save_model_outputs(paths.textcnn_output, textcnn_model, test_frame, textcnn_predictions, id_to_label, test_metadata, {"method": "Unicode word regex", "pattern": "\\w+", "max_length": TEXTCNN_MAX_LENGTH, "padding": "right pad with <PAD>", "truncation": "right truncate"})
        rows.append({
            "model_id": textcnn_model["model_id"], "model_family": textcnn_model["model_family"], "condition": "baseline", "seed": textcnn_model["seed"],
            "msa_validation_macro_f1": textcnn_model["msa_validation_metrics"]["macro_f1"],
            "saudi_macro_f1": textcnn_output["metrics"]["macro_f1"], "saudi_weighted_f1": textcnn_output["metrics"]["weighted_f1"], "saudi_accuracy": textcnn_output["metrics"]["accuracy"],
        })
        first_tokenizer = AutoTokenizer.from_pretrained(paths.arabert_checkpoint("E0", 42), use_fast=True)
        arabert_tokens, arabert_preprocessing = prepare_arabert(test_frame["text"].tolist(), first_tokenizer)
        for metadata in arabert_models:
            prediction = predict_arabert(Path(metadata["checkpoint"]), arabert_tokens, label_to_id)
            output = save_model_outputs(paths.arabert_saudi_output(metadata["condition"], metadata["seed"]), metadata, test_frame, prediction, id_to_label, test_metadata, arabert_preprocessing)
            rows.append({
                "model_id": metadata["model_id"], "model_family": metadata["model_family"], "condition": metadata["condition"], "seed": metadata["seed"],
                "msa_validation_macro_f1": metadata["msa_validation_metrics"]["macro_f1"],
                "saudi_macro_f1": output["metrics"]["macro_f1"], "saudi_weighted_f1": output["metrics"]["weighted_f1"], "saudi_accuracy": output["metrics"]["accuracy"],
            })
        seed_summary, summary = make_summary(rows)
        seed_summary.to_csv(paths.result_root / "final_saudi_seed_results.csv", index=False)
        summary.to_csv(paths.result_root / "final_saudi_model_comparison.csv", index=False)
        manifest = {
            "owner": OWNER,
            "status": "completed",
            "completed_at_utc": utc_now(),
            "evaluation_type": "one_time_final_held_out_saudi_evaluation",
            "primary_metric": "macro_f1",
            "test_access_metadata": test_metadata,
            "models_evaluated": {"svm": 1, "textcnn": 1, "arabert": len(arabert_models), "total": len(rows)},
            "outputs": {"seed_results": "final_saudi_seed_results.csv", "model_comparison": "final_saudi_model_comparison.csv"},
            "saudi_test_accessed": True,
        }
        write_json(paths.manifest_path, manifest)
        paths.lock_path.unlink()
        return manifest
    except Exception:
        failure = read_json(paths.lock_path)
        failure.update({"status": "failed", "failed_at_utc": utc_now(), "saudi_test_accessed": test_opened, "failure_traceback": traceback.format_exc()})
        write_json(paths.lock_path, failure)
        raise


def resolve_pretest_path_failure(project_root: str | Path) -> dict[str, Any]:
    """Archive only the documented pre-read missing-path failure; never read the test."""
    paths = Paths(Path(project_root))
    assert not paths.manifest_path.exists(), "Completed final evaluation exists; recovery is not allowed."
    assert paths.lock_path.is_file(), "No evaluation lock exists to recover."
    lock = read_json(paths.lock_path)
    failure_traceback = str(lock.get("failure_traceback", ""))
    assert lock.get("status") == "failed", "Recovery applies only to a failed run."
    assert "Missing Saudi test:" in failure_traceback, "Recovery is allowed only for the missing-file pre-read failure."
    assert "saudi_test.csv" in failure_traceback, "Recovery is not applicable to another failure."
    assert not any(paths.arabert_saudi_output(condition, seed).exists() for condition in CONDITIONS for seed in SEEDS)
    assert not paths.svm_output.exists() and not paths.textcnn_output.exists()
    archived = paths.result_root / "final_saudi_evaluation_pretest_path_failure_2026-09-14.json"
    assert not archived.exists(), f"Recovery record already exists: {archived}"
    record = {
        "owner": OWNER,
        "status": "recovered_pretest_path_failure",
        "recovery_at_utc": utc_now(),
        "reason": "The previous official command stopped at a missing legacy file path before pd.read_csv, model inference, metrics, or output creation.",
        "previous_lock": lock,
        "saudi_test_accessed": False,
        "outputs_created": False,
        "next_step": "Run preflight again, then run the corrected one-time official evaluation.",
    }
    write_json(archived, record)
    paths.lock_path.unlink()
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--mode", choices=("preflight", "official", "resolve_pretest_path_failure"), required=True)
    args = parser.parse_args()
    if args.mode == "preflight":
        result = preflight(args.project_root)
    elif args.mode == "official":
        result = official(args.project_root)
    else:
        result = resolve_pretest_path_failure(args.project_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
