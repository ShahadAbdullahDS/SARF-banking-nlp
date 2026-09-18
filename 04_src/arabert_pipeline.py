"""Run one clean, frozen SARF AraBERT condition from project Drive.

Required Drive inputs:
- 02_configs/arabert_config.json
- 02_configs/arabert_label_mapping.json
- 02_processed_data/msa_train_v1.csv
- 02_processed_data/msa_val_v1.csv
- E1, E2, E3, or EB only: the assigned v2 CPT corpus or corpora named in arabert_config.json

The script does not load, name, or construct a path to the held-out Saudi test.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from arabert.preprocess import ArabertPreprocessor
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

OWNER = "A"
OFFICIAL_SEEDS = (42, 123, 2026)
NUM_LABELS = 77
TRAIN_ROWS = 10_732
VALIDATION_ROWS = 1_229


@dataclass(frozen=True)
class ProjectPaths:
    """The fixed clean Drive layout for final AraBERT E0, E1, E2, E3, and EB."""

    root: Path

    @property
    def config(self) -> Path:
        return self.root / "02_configs" / "arabert_config.json"

    @property
    def label_mapping(self) -> Path:
        return self.root / "02_configs" / "arabert_label_mapping.json"

    @property
    def run_root(self) -> Path:
        return self.root / "05_runs" / "arabert"

    @property
    def model_root(self) -> Path:
        return self.root / "06_models_and_checkpoints" / "arabert"

    def run_dir(self, condition: str, seed: int) -> Path:
        return self.run_root / condition / f"seed_{seed}"

    def model_dir(self, condition: str, seed: int) -> Path:
        return self.model_root / condition / f"seed_{seed}"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_gpu() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required; do not run full AraBERT training on CPU.")


def set_all_seeds(seed: int) -> None:
    """Set all controllable random states before a model-training stage."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(seed)


def environment_record() -> dict[str, str]:
    gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    return {
        "python": sys.version.replace("\n", " "),
        "torch": torch.__version__,
        "transformers": version("transformers"),
        "accelerate": version("accelerate"),
        "datasets": version("datasets"),
        "arabert": version("arabert"),
        "scikit_learn": version("scikit-learn"),
        "gpu": gpu,
    }


def validate_config(config: dict[str, Any], condition: str, seed: int) -> None:
    """Reject accidental changes to the frozen clean five-condition procedure."""
    expected_fine_tuning = {
        "max_length": 128,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 32,
        "gradient_accumulation_steps": 4,
        "effective_train_batch_size": 64,
        "learning_rate": 2e-5,
        "num_train_epochs": 20,
        "warmup_ratio": 0.0,
        "fp16": True,
        "early_stopping_patience": 3,
    }
    expected_checkpoint_selection = {
        "metric": "macro_f1 on MSA validation",
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "greater_is_better": True,
    }

    assert config["owner"] == OWNER
    assert config["base_checkpoint"] == "aubmindlab/bert-base-arabertv2"
    assert config["official_conditions"] == ["E0", "E1", "E2", "E3", "EB"]
    assert config["official_seeds"] == list(OFFICIAL_SEEDS)
    assert condition in {"E0", "E1", "E2", "E3", "EB"}
    assert seed in OFFICIAL_SEEDS
    assert config["data_policy"]["train_relative_path"] == "02_processed_data/msa_train_v1.csv"
    assert config["data_policy"]["validation_relative_path"] == "02_processed_data/msa_val_v1.csv"
    assert config["data_policy"]["label_mapping_file"] == "02_configs/arabert_label_mapping.json"
    assert config["data_policy"]["full_train_rows"] == TRAIN_ROWS
    assert config["data_policy"]["validation_rows"] == VALIDATION_ROWS
    assert config["data_policy"]["num_labels"] == NUM_LABELS
    assert config["fine_tuning"] == expected_fine_tuning
    assert config["checkpoint_selection"] == expected_checkpoint_selection
    assert config["cpt_defaults"] == {
        "text_column_only": True,
        "objective": "masked_language_modeling",
        "num_train_epochs": 1,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "learning_rate": 5e-5,
        "mlm_probability": 0.15,
        "max_length": 128,
        "save_strategy": "no",
    }
    conditions = config["conditions"]
    assert set(conditions) == {"E0", "E1", "E2", "E3", "EB"}
    assert conditions["E0"]["cpt_enabled"] is False
    assert conditions["E0"]["cpt"] is None
    assert conditions["E1"]["cpt_enabled"] is True
    assert conditions["E2"]["cpt_enabled"] is True
    assert conditions["E3"]["cpt_enabled"] is True
    assert conditions["EB"]["cpt_enabled"] is True
    assert conditions["E1"]["cpt"]["corpus_files"] == ["general_3k_final_v2.csv"]
    assert conditions["E2"]["cpt"]["corpus_files"] == ["general_6k_final_v2.csv"]
    assert conditions["E3"]["cpt"]["corpus_files"] == ["general_12k_final_v2.csv"]
    assert conditions["EB"]["cpt"]["corpus_files"] == [
        "general_12k_final_v2.csv",
        "banking_3k_final_v2.csv",
    ]


def load_mapping(paths: ProjectPaths) -> tuple[dict[str, int], dict[int, str]]:
    """Load the one frozen global mapping used across every condition and seed."""
    assert paths.label_mapping.is_file(), f"Missing global label mapping: {paths.label_mapping}"
    mapping = read_json(paths.label_mapping)
    assert mapping["owner"] == OWNER
    assert mapping["num_labels"] == NUM_LABELS
    label_to_id = mapping["label_to_id"]
    id_to_label = {int(key): value for key, value in mapping["id_to_label"].items()}
    assert len(label_to_id) == NUM_LABELS
    assert len(id_to_label) == NUM_LABELS
    assert set(label_to_id.values()) == set(range(NUM_LABELS))
    assert set(id_to_label) == set(range(NUM_LABELS))
    for label, label_id in label_to_id.items():
        assert id_to_label[label_id] == label
    return label_to_id, id_to_label


def load_msa_data(
    paths: ProjectPaths,
    config: dict[str, Any],
    label_to_id: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Load only the approved labelled MSA training and validation files."""
    train_path = paths.root / config["data_policy"]["train_relative_path"]
    validation_path = paths.root / config["data_policy"]["validation_relative_path"]
    assert train_path.is_file(), f"Missing MSA training input: {train_path}"
    assert validation_path.is_file(), f"Missing MSA validation input: {validation_path}"

    train_df = pd.read_csv(train_path)
    validation_df = pd.read_csv(validation_path)
    for split_name, frame, expected_rows in (
        ("train", train_df, TRAIN_ROWS),
        ("validation", validation_df, VALIDATION_ROWS),
    ):
        assert {"label", "text"}.issubset(frame.columns), (
            f"{split_name} requires label and text columns"
        )
        assert len(frame) == expected_rows, f"Unexpected {split_name} row count: {len(frame)}"
        assert frame["label"].notna().all(), f"{split_name} has null labels"
        assert frame["text"].notna().all(), f"{split_name} has null text"
        assert frame["text"].astype(str).str.strip().ne("").all(), (
            f"{split_name} has blank text"
        )

    train_df = train_df.copy()
    validation_df = validation_df.copy()
    train_df["label"] = train_df["label"].astype(str)
    validation_df["label"] = validation_df["label"].astype(str)
    assert not (set(validation_df["label"]) - set(label_to_id))
    assert not (set(train_df["label"]) - set(label_to_id))

    train_df.insert(0, "source_row_id", np.arange(len(train_df), dtype=int))
    validation_df.insert(0, "source_row_id", np.arange(len(validation_df), dtype=int))
    train_df["labels"] = train_df["label"].map(label_to_id).astype(int)
    validation_df["labels"] = validation_df["label"].map(label_to_id).astype(int)

    input_access_log = [
        {
            "role": "fine_tuning_train",
            "relative_path": config["data_policy"]["train_relative_path"],
            "rows_loaded": TRAIN_ROWS,
        },
        {
            "role": "validation",
            "relative_path": config["data_policy"]["validation_relative_path"],
            "rows_loaded": VALIDATION_ROWS,
        },
    ]
    return train_df, validation_df, input_access_log


def prepare_and_tokenize_msa(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    tokenizer: Any,
    base_checkpoint: str,
    max_length: int,
) -> tuple[Dataset, Dataset, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Apply frozen MSA preprocessing, audit lengths, and tokenize MSA text."""
    preprocessor = ArabertPreprocessor(model_name=base_checkpoint)
    train_df["prepared_text"] = train_df["text"].map(
        lambda text: preprocessor.preprocess(str(text))
    )
    validation_df["prepared_text"] = validation_df["text"].map(
        lambda text: preprocessor.preprocess(str(text))
    )
    assert train_df["prepared_text"].str.strip().ne("").all()
    assert validation_df["prepared_text"].str.strip().ne("").all()

    all_text = train_df["prepared_text"].tolist() + validation_df["prepared_text"].tolist()
    untruncated_ids = tokenizer(all_text, add_special_tokens=True, truncation=False)["input_ids"]
    lengths = np.array([len(ids) for ids in untruncated_ids])
    train_lengths = lengths[: len(train_df)]
    validation_lengths = lengths[len(train_df) :]
    truncation_audit = {
        "max_length": max_length,
        "train_truncation_rate": float(np.mean(train_lengths > max_length)),
        "validation_truncation_rate": float(np.mean(validation_lengths > max_length)),
        "train_max_untruncated_length": int(train_lengths.max()),
        "validation_max_untruncated_length": int(validation_lengths.max()),
    }
    assert truncation_audit["train_truncation_rate"] == 0.0
    assert truncation_audit["validation_truncation_rate"] == 0.0

    def tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(batch["prepared_text"], truncation=True, max_length=max_length)

    train_dataset = Dataset.from_pandas(
        train_df[["prepared_text", "labels"]], preserve_index=False
    ).map(tokenize, batched=True, remove_columns=["prepared_text"])
    validation_dataset = Dataset.from_pandas(
        validation_df[["prepared_text", "labels"]], preserve_index=False
    ).map(tokenize, batched=True, remove_columns=["prepared_text"])
    return train_dataset, validation_dataset, train_df, validation_df, truncation_audit


class TimedTrainer(Trainer):
    """Record validation and checkpoint time separately for the runtime evidence."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.runtime_breakdown = {
            "validation_during_training_seconds": 0.0,
            "checkpoint_saving_during_training_seconds": 0.0,
        }
        super().__init__(*args, **kwargs)

    def evaluate(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        start = time.perf_counter()
        result = super().evaluate(*args, **kwargs)
        self.runtime_breakdown["validation_during_training_seconds"] += (
            time.perf_counter() - start
        )
        return result

    def _save_checkpoint(self, *args: Any, **kwargs: Any) -> None:
        start = time.perf_counter()
        result = super()._save_checkpoint(*args, **kwargs)
        self.runtime_breakdown["checkpoint_saving_during_training_seconds"] += (
            time.perf_counter() - start
        )
        return result


def compute_metrics(prediction: Any) -> dict[str, float]:
    logits, true_ids = prediction
    if isinstance(logits, tuple):
        logits = logits[0]
    predicted_ids = np.argmax(logits, axis=-1)
    labels = np.arange(NUM_LABELS)
    return {
        "macro_f1": f1_score(
            true_ids, predicted_ids, labels=labels, average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(
            true_ids, predicted_ids, labels=labels, average="weighted", zero_division=0
        ),
        "accuracy": accuracy_score(true_ids, predicted_ids),
    }


def run_cpt(
    paths: ProjectPaths,
    config: dict[str, Any],
    condition: str,
    tokenizer: Any,
    seed: int,
    local_work_dir: Path,
) -> tuple[Path, float, dict[str, Any]]:
    """Run one MLM CPT epoch over the exact v2 corpus assignment for a condition."""
    condition_spec = config["conditions"][condition]
    assert condition_spec["cpt_enabled"] is True
    cpt_assignment = condition_spec["cpt"]
    cpt_settings = config["cpt_defaults"]
    manifest_path = paths.root / cpt_assignment["manifest_relative_path"]
    corpus_dir = paths.root / cpt_assignment["corpus_directory_relative_path"]
    corpus_files = cpt_assignment["corpus_files"]
    assert manifest_path.is_file(), f"Missing CPT manifest: {manifest_path}"
    assert corpus_dir.is_dir(), f"Missing CPT corpus directory: {corpus_dir}"

    start = time.perf_counter()
    manifest_text = manifest_path.read_text(encoding="utf-8")
    frames: list[pd.DataFrame] = []
    observed_rows: dict[str, int] = {}
    for file_name, expected_rows in zip(
        corpus_files, cpt_assignment["corpus_file_rows"], strict=True
    ):
        assert file_name.endswith("_final_v2.csv"), f"Non-v2 CPT input rejected: {file_name}"
        assert file_name in manifest_text, f"CPT file is absent from manifest: {file_name}"
        corpus_path = corpus_dir / file_name
        assert corpus_path.is_file(), f"Missing approved CPT input: {corpus_path}"
        frame = pd.read_csv(corpus_path)
        assert "text" in frame.columns, f"CPT input lacks a text column: {corpus_path}"
        assert len(frame) == expected_rows, (
            f"Unexpected row count for {file_name}: {len(frame)}"
        )
        assert frame["text"].notna().all(), f"Null CPT text in {file_name}"
        frame = frame[["text"]].copy()
        frame["text"] = frame["text"].astype(str)
        assert frame["text"].str.strip().ne("").all(), f"Blank CPT text in {file_name}"
        frames.append(frame)
        observed_rows[file_name] = len(frame)

    cpt_df = pd.concat(frames, ignore_index=True)
    assert len(cpt_df) == cpt_assignment["total_rows"]

    dataset = Dataset.from_pandas(cpt_df, preserve_index=False)
    tokenized = dataset.map(
        lambda batch: tokenizer(
            batch["text"], truncation=True, max_length=cpt_settings["max_length"]
        ),
        batched=True,
        remove_columns=["text"],
        desc=f"Tokenizing approved {condition} CPT corpus",
    )

    cpt_work_dir = local_work_dir / "cpt"
    cpt_checkpoint = cpt_work_dir / "checkpoint"
    model = AutoModelForMaskedLM.from_pretrained(config["base_checkpoint"])
    arguments = TrainingArguments(
        output_dir=str(cpt_work_dir),
        overwrite_output_dir=False,
        num_train_epochs=cpt_settings["num_train_epochs"],
        per_device_train_batch_size=cpt_settings["per_device_train_batch_size"],
        gradient_accumulation_steps=cpt_settings["gradient_accumulation_steps"],
        learning_rate=cpt_settings["learning_rate"],
        fp16=True,
        logging_steps=100,
        save_strategy=cpt_settings["save_strategy"],
        report_to=[],
        seed=seed,
        data_seed=seed,
        remove_unused_columns=True,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=cpt_settings["mlm_probability"],
        ),
        tokenizer=tokenizer,
    )
    train_result = trainer.train()
    assert math.isfinite(float(train_result.training_loss))
    cpt_checkpoint.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(cpt_checkpoint))
    assert (cpt_checkpoint / "config.json").is_file()
    return cpt_checkpoint, time.perf_counter() - start, {
        "condition": condition,
        "corpus_files": corpus_files,
        "corpus_file_rows": observed_rows,
        "total_rows": len(cpt_df),
        "text_column_only": True,
        "objective": "masked_language_modeling",
        "epochs": cpt_settings["num_train_epochs"],
        "training_loss": float(train_result.training_loss),
    }


def run_cpt_technical_check(project_root: str | Path) -> dict[str, Any]:
    """Verify the E1 v2 CPT path using 128 rows and five MLM steps only.

    This is not an official E1 run.  The temporary checkpoint is saved and
    reloaded locally to prove the checkpoint path works, then removed; no
    technical-check model is retained on Drive.
    """
    run_id = "cpt_technical_check_v2"
    seed = 42
    subset_rows = 128
    max_steps = 5

    require_gpu()
    paths = ProjectPaths(Path(project_root))
    assert paths.root.is_dir(), f"Project root not found: {paths.root}"
    assert paths.config.is_file(), f"Missing clean AraBERT config: {paths.config}"
    config = read_json(paths.config)
    validate_config(config, "E1", seed)

    e1_cpt = config["conditions"]["E1"]["cpt"]
    assert e1_cpt["corpus_files"] == ["general_3k_final_v2.csv"]
    assert e1_cpt["total_rows"] == 3_000
    corpus_dir = paths.root / e1_cpt["corpus_directory_relative_path"]
    corpus_path = corpus_dir / e1_cpt["corpus_files"][0]
    manifest_path = paths.root / e1_cpt["manifest_relative_path"]
    assert corpus_path.is_file(), f"Missing approved CPT input: {corpus_path}"
    assert manifest_path.is_file(), f"Missing CPT manifest: {manifest_path}"
    assert corpus_path.name in manifest_path.read_text(encoding="utf-8")

    run_dir = paths.run_root / "dev" / "cpt_technical_check"
    local_work_dir = Path(config["artifact_storage"]["temporary_trainer_work_root"]) / (
        "sarf_cpt_technical_check_work"
    )
    assert not run_dir.exists(), f"Refusing to overwrite technical check: {run_dir}"
    if local_work_dir.exists():
        shutil.rmtree(local_work_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    local_work_dir.mkdir(parents=True, exist_ok=False)

    try:
        start = time.perf_counter()
        set_all_seeds(seed)
        corpus_df = pd.read_csv(corpus_path)
        assert "text" in corpus_df.columns
        assert len(corpus_df) == 3_000
        assert corpus_df["text"].notna().all()
        subset_df = corpus_df.head(subset_rows)[["text"]].copy()
        subset_df["text"] = subset_df["text"].astype(str)
        assert subset_df["text"].str.strip().ne("").all()
        corpus_loading_seconds = time.perf_counter() - start

        tokenizer = AutoTokenizer.from_pretrained(config["base_checkpoint"], use_fast=True)
        dataset = Dataset.from_pandas(subset_df, preserve_index=False)
        tokenized = dataset.map(
            lambda batch: tokenizer(
                batch["text"],
                truncation=True,
                max_length=config["cpt_defaults"]["max_length"],
            ),
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing fixed E1 CPT technical subset",
        )
        model = AutoModelForMaskedLM.from_pretrained(config["base_checkpoint"])
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=str(local_work_dir),
                overwrite_output_dir=False,
                max_steps=max_steps,
                per_device_train_batch_size=config["cpt_defaults"][
                    "per_device_train_batch_size"
                ],
                gradient_accumulation_steps=config["cpt_defaults"][
                    "gradient_accumulation_steps"
                ],
                learning_rate=config["cpt_defaults"]["learning_rate"],
                fp16=config["fine_tuning"]["fp16"],
                logging_steps=1,
                save_strategy="steps",
                save_steps=max_steps,
                save_total_limit=1,
                report_to=[],
                seed=seed,
                data_seed=seed,
                remove_unused_columns=True,
            ),
            train_dataset=tokenized,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=True,
                mlm_probability=config["cpt_defaults"]["mlm_probability"],
            ),
            tokenizer=tokenizer,
        )
        training_start = time.perf_counter()
        train_result = trainer.train()
        mlm_training_seconds = time.perf_counter() - training_start
        assert math.isfinite(float(train_result.training_loss))

        checkpoint_dir = local_work_dir / "checkpoint"
        trainer.save_model(str(checkpoint_dir))
        assert (checkpoint_dir / "config.json").is_file()
        assert (checkpoint_dir / "tokenizer_config.json").is_file()
        reloaded_tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        reloaded_model = AutoModelForMaskedLM.from_pretrained(checkpoint_dir).to("cuda").eval()
        probe = reloaded_tokenizer(
            subset_df["text"].head(8).tolist(),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config["cpt_defaults"]["max_length"],
        )
        with torch.no_grad():
            probe_logits = reloaded_model(
                **{name: tensor.to("cuda") for name, tensor in probe.items()}
            ).logits
        assert probe_logits.shape[0] == 8
        checkpoint_reload_passed = True

        summary = {
            "run_id": run_id,
            "owner": OWNER,
            "status": "completed",
            "run_type": "development_cpt_technical_check",
            "condition": "CPT_TECHNICAL_CHECK",
            "base_checkpoint": config["base_checkpoint"],
            "corpus_file": corpus_path.name,
            "corpus_relative_path": str(corpus_path.relative_to(paths.root)),
            "corpus_sha256": sha256(corpus_path),
            "manifest_relative_path": str(manifest_path.relative_to(paths.root)),
            "subset_selection": "first_128_rows",
            "subset_rows": subset_rows,
            "max_length": config["cpt_defaults"]["max_length"],
            "max_steps": max_steps,
            "mlm_probability": config["cpt_defaults"]["mlm_probability"],
            "text_column_only": True,
            "labels_loaded": False,
            "train_loss": float(train_result.training_loss),
            "runtime_seconds": {
                "corpus_loading_seconds": corpus_loading_seconds,
                "mlm_training_seconds": mlm_training_seconds,
                "total_runtime_seconds": time.perf_counter() - start,
            },
            "checks": {
                "v2_manifest_verified": True,
                "approved_general_3k_v2_only": True,
                "mlm_objective_executed": True,
                "finite_mlm_loss": True,
                "checkpoint_saved_locally": True,
                "checkpoint_reloaded_locally": checkpoint_reload_passed,
                "checkpoint_retained_on_drive": False,
            },
            "saudi_test_accessed": False,
            "not_an_official_e1_run": True,
        }
        write_json(run_dir / "evaluation_summary.json", summary)
        write_json(
            run_dir / "input_access_log.json",
            [
                {
                    "role": "CPT technical check",
                    "relative_path": str(corpus_path.relative_to(paths.root)),
                    "rows_loaded": 3_000,
                    "rows_used": subset_rows,
                    "columns_used": ["text"],
                }
            ],
        )
        (run_dir / "environment.txt").write_text(
            "\n".join(f"{key}={value}" for key, value in environment_record().items())
            + "\n",
            encoding="utf-8",
        )
        (run_dir / "run_log.md").write_text(
            "\n".join(
                [
                    f"# {run_id}",
                    "",
                    f"- Owner: {OWNER}",
                    "- Type: development CPT technical check; not an official E1 run",
                    f"- Corpus: {corpus_path.name} only",
                    f"- Subset: first {subset_rows} rows; text column only",
                    f"- MLM steps: {max_steps}",
                    f"- MLM training loss: {float(train_result.training_loss):.6f}",
                    "- Local checkpoint save and reload: passed",
                    "- Technical checkpoint retained on Drive: no",
                    "- Saudi test accessed: false",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return summary
    except Exception:
        (run_dir / "failure_traceback.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        raise
    finally:
        if local_work_dir.exists():
            shutil.rmtree(local_work_dir)
        gc.collect()
        torch.cuda.empty_cache()


def save_final_outputs(
    run_dir: Path,
    model_dir: Path,
    trainer: TimedTrainer,
    tokenizer: Any,
    validation_dataset: Dataset,
    validation_df: pd.DataFrame,
    id_to_label: dict[int, str],
) -> tuple[dict[str, float], float, float]:
    """Export metrics, predictions, error analysis, and reload verification."""
    start = time.perf_counter()
    history = pd.DataFrame(trainer.state.log_history)
    history.to_csv(run_dir / "train_eval_history.csv", index=False)
    evaluation_history = history.loc[history["eval_macro_f1"].notna()].copy()
    best_row = evaluation_history.loc[evaluation_history["eval_macro_f1"].idxmax()]
    best_epoch = float(best_row["epoch"])
    best_macro_f1 = float(best_row["eval_macro_f1"])
    assert np.isclose(float(trainer.state.best_metric), best_macro_f1)

    best_checkpoint = model_dir / "best_checkpoint"
    trainer.save_model(str(best_checkpoint))
    assert (best_checkpoint / "tokenizer_config.json").is_file()

    prediction = trainer.predict(validation_dataset, metric_key_prefix="validation")
    logits = (
        prediction.predictions[0]
        if isinstance(prediction.predictions, tuple)
        else prediction.predictions
    )
    true_ids = prediction.label_ids
    predicted_ids = np.argmax(logits, axis=-1)
    assert logits.shape == (VALIDATION_ROWS, NUM_LABELS)
    assert np.unique(predicted_ids).size > 1
    metrics = compute_metrics((logits, true_ids))
    assert np.isclose(metrics["macro_f1"], best_macro_f1)

    labels = [id_to_label[index] for index in range(NUM_LABELS)]
    predictions_df = pd.DataFrame(
        {
            "row_id": validation_df["source_row_id"].to_numpy(),
            "text": validation_df["text"].to_numpy(),
            "true_label": validation_df["label"].to_numpy(),
            "predicted_label": [id_to_label[int(value)] for value in predicted_ids],
        }
    )
    predictions_df["is_correct"] = (
        predictions_df["true_label"] == predictions_df["predicted_label"]
    )
    predictions_df.to_csv(run_dir / "validation_predictions.csv", index=False)

    report = classification_report(
        true_ids,
        predicted_ids,
        labels=np.arange(NUM_LABELS),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(
        [
            {
                "label": label,
                "label_id": index,
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1_score": report[label]["f1-score"],
                "support": report[label]["support"],
            }
            for index, label in enumerate(labels)
        ]
    ).to_csv(run_dir / "per_class_metrics.csv", index=False)

    matrix = confusion_matrix(true_ids, predicted_ids, labels=np.arange(NUM_LABELS))
    matrix_df = pd.DataFrame(matrix, index=labels, columns=labels)
    matrix_df.index.name = "true_label"
    matrix_df.columns.name = "predicted_label"
    matrix_df.to_csv(run_dir / "confusion_matrix.csv")
    write_json(
        run_dir / "validation_metrics.json",
        {
            "owner": OWNER,
            "split": "msa_val_v1",
            "best_epoch": best_epoch,
            "metrics": metrics,
        },
    )

    reloaded_tokenizer = AutoTokenizer.from_pretrained(best_checkpoint)
    reloaded_model = (
        AutoModelForSequenceClassification.from_pretrained(best_checkpoint)
        .to("cuda")
        .eval()
    )
    probe = reloaded_tokenizer(
        validation_df["prepared_text"].head(8).tolist(),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )
    with torch.no_grad():
        reloaded_logits = reloaded_model(
            **{name: tensor.to("cuda") for name, tensor in probe.items()}
        ).logits
    np.testing.assert_array_equal(
        reloaded_logits.argmax(dim=-1).cpu().numpy(), predicted_ids[:8]
    )
    return metrics, best_epoch, time.perf_counter() - start


def run_official(project_root: str | Path, condition: str, seed: int) -> dict[str, Any]:
    """Run one fresh E0, E1, E2, E3, or EB seed without overwriting outputs."""
    condition = condition.upper()
    if condition not in {"E0", "E1", "E2", "E3", "EB"}:
        raise ValueError("Condition must be one of E0, E1, E2, E3, or EB.")
    if seed not in OFFICIAL_SEEDS:
        raise ValueError(f"Seed must be one of {OFFICIAL_SEEDS}.")

    require_gpu()
    paths = ProjectPaths(Path(project_root))
    assert paths.root.is_dir(), f"Project root not found: {paths.root}"
    assert paths.config.is_file(), f"Missing clean AraBERT config: {paths.config}"
    config = read_json(paths.config)
    validate_config(config, condition, seed)
    label_to_id, id_to_label = load_mapping(paths)
    condition_spec = config["conditions"][condition]
    cpt_files = (
        condition_spec["cpt"]["corpus_files"]
        if condition_spec["cpt_enabled"]
        else []
    )

    run_id = f"{condition}_seed_{seed}"
    run_dir = paths.run_dir(condition, seed)
    model_dir = paths.model_dir(condition, seed)
    local_work_dir = Path(config["artifact_storage"]["temporary_trainer_work_root"]) / (
        f"sarf_{run_id}_work"
    )

    assert not run_dir.exists(), f"Refusing to overwrite existing run: {run_dir}"
    assert not model_dir.exists(), f"Refusing to overwrite existing model: {model_dir}"
    if local_work_dir.exists():
        shutil.rmtree(local_work_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    model_dir.mkdir(parents=True, exist_ok=False)
    local_work_dir.mkdir(parents=True, exist_ok=False)

    run_config = {
        "run_id": run_id,
        "owner": OWNER,
        "model_family": "AraBERTv2-base",
        "condition": condition,
        "run_type": "official",
        "seed": seed,
        "base_checkpoint": config["base_checkpoint"],
        "arabert_config_file": paths.config.name,
        "arabert_config_sha256": sha256(paths.config),
        "label_mapping_file": paths.label_mapping.name,
        "label_mapping_sha256": sha256(paths.label_mapping),
        "train_split": config["data_policy"]["train_split"],
        "validation_split": config["data_policy"]["validation_split"],
        "fine_tuning": config["fine_tuning"],
        "preprocessing": config["preprocessing"],
        "checkpoint_selection": config["checkpoint_selection"],
        "cpt": {
            "enabled": condition_spec["cpt_enabled"],
            "corpus_files": cpt_files,
            "total_rows": (
                condition_spec["cpt"]["total_rows"]
                if condition_spec["cpt_enabled"]
                else 0
            ),
        },
        "status": "running",
    }
    write_json(run_dir / "config.json", run_config)
    (run_dir / "environment.txt").write_text(
        "\n".join(f"{key}={value}" for key, value in environment_record().items())
        + "\n",
        encoding="utf-8",
    )

    timings: dict[str, float | str] = {}
    try:
        start = time.perf_counter()
        set_all_seeds(seed)
        train_df, validation_df, input_access_log = load_msa_data(
            paths, config, label_to_id
        )
        tokenizer = AutoTokenizer.from_pretrained(config["base_checkpoint"], use_fast=True)
        train_dataset, validation_dataset, train_df, validation_df, truncation_audit = (
            prepare_and_tokenize_msa(
                train_df,
                validation_df,
                tokenizer,
                config["base_checkpoint"],
                config["fine_tuning"]["max_length"],
            )
        )
        timings["loading_and_tokenization_seconds"] = time.perf_counter() - start
        write_json(run_dir / "input_access_log.json", input_access_log)

        cpt_checkpoint: Path | None = None
        cpt_details: dict[str, Any] | None = None
        if condition_spec["cpt_enabled"]:
            cpt_checkpoint, cpt_seconds, cpt_details = run_cpt(
                paths, config, condition, tokenizer, seed, local_work_dir
            )
            timings["cpt_seconds"] = cpt_seconds

        set_all_seeds(seed)
        torch.cuda.empty_cache()
        model_source = cpt_checkpoint or config["base_checkpoint"]
        model = AutoModelForSequenceClassification.from_pretrained(
            model_source,
            num_labels=NUM_LABELS,
            label2id=label_to_id,
            id2label=id_to_label,
        )
        assert model.config.num_labels == NUM_LABELS
        arguments = TrainingArguments(
            output_dir=str(local_work_dir / "fine_tuning"),
            overwrite_output_dir=False,
            learning_rate=config["fine_tuning"]["learning_rate"],
            per_device_train_batch_size=config["fine_tuning"]["per_device_train_batch_size"],
            per_device_eval_batch_size=config["fine_tuning"]["per_device_eval_batch_size"],
            gradient_accumulation_steps=config["fine_tuning"]["gradient_accumulation_steps"],
            num_train_epochs=config["fine_tuning"]["num_train_epochs"],
            warmup_ratio=config["fine_tuning"]["warmup_ratio"],
            fp16=config["fine_tuning"]["fp16"],
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=80,
            logging_first_step=True,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            save_total_limit=config["artifact_storage"]["temporary_checkpoint_limit"],
            seed=seed,
            data_seed=seed,
            report_to=[],
            remove_unused_columns=True,
        )
        trainer = TimedTrainer(
            model=model,
            args=arguments,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_metrics,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=config["fine_tuning"]["early_stopping_patience"]
                )
            ],
        )
        fine_tuning_start = time.perf_counter()
        trainer.train()
        training_wall_seconds = time.perf_counter() - fine_tuning_start
        assert trainer.state.best_model_checkpoint is not None
        assert trainer.state.best_metric is not None

        metrics, best_epoch, finalization_seconds = save_final_outputs(
            run_dir,
            model_dir,
            trainer,
            tokenizer,
            validation_dataset,
            validation_df,
            id_to_label,
        )
        validation_seconds = trainer.runtime_breakdown[
            "validation_during_training_seconds"
        ]
        checkpoint_seconds = trainer.runtime_breakdown[
            "checkpoint_saving_during_training_seconds"
        ]
        timings.update(
            {
                "fine_tuning_seconds": max(
                    training_wall_seconds - validation_seconds - checkpoint_seconds, 0.0
                ),
                "validation_during_training_seconds": validation_seconds,
                "checkpoint_saving_during_training_seconds": checkpoint_seconds,
                "final_validation_export_reload_seconds": finalization_seconds,
                "validation_and_checkpoint_saving_seconds": (
                    validation_seconds + checkpoint_seconds + finalization_seconds
                ),
            }
        )
        timings["total_runtime_seconds"] = float(
            timings["loading_and_tokenization_seconds"]
            + timings.get("cpt_seconds", 0.0)
            + timings["fine_tuning_seconds"]
            + timings["validation_and_checkpoint_saving_seconds"]
        )
        timings["total_runtime_definition"] = (
            "Sum of measured loading/tokenization, CPT when applicable, fine-tuning, "
            "validation, checkpoint saving, export, and reload steps."
        )

        summary = {
            "run_id": run_id,
            "owner": OWNER,
            "model_family": "AraBERTv2-base",
            "condition": condition,
            "run_type": "official",
            "seed": seed,
            "base_checkpoint": config["base_checkpoint"],
            "config_id": config["config_id"],
            "train_split": config["data_policy"]["train_split"],
            "validation_split": config["data_policy"]["validation_split"],
            "train_rows": TRAIN_ROWS,
            "validation_rows": VALIDATION_ROWS,
            "num_labels": NUM_LABELS,
            "cpt_corpus_files": cpt_files,
            "cpt_details": cpt_details,
            "checkpoint_selection_metric": "macro_f1 on MSA validation",
            "best_epoch": best_epoch,
            "metrics": metrics,
            "prediction_file": "validation_predictions.csv",
            "per_class_file": "per_class_metrics.csv",
            "confusion_matrix_file": "confusion_matrix.csv",
            "runtime_seconds": timings,
            "truncation_audit": truncation_audit,
            "checks": {
                "labels_77_verified": True,
                "logits_shape_verified": [VALIDATION_ROWS, NUM_LABELS],
                "predictions_not_single_label": True,
                "best_checkpoint_selected_by_macro_f1": True,
                "checkpoint_reload_passed": True,
                "approved_msa_inputs_only": True,
                "zero_truncation_at_frozen_max_length": True,
                "fresh_base_checkpoint_used": True,
            },
            "saudi_test_accessed": False,
            "status": "completed",
        }
        write_json(run_dir / "evaluation_summary.json", summary)

        run_config.update(
            {
                "status": "completed",
                "best_epoch": best_epoch,
                "best_validation_macro_f1": metrics["macro_f1"],
                "runtime_seconds": timings,
            }
        )
        write_json(run_dir / "config.json", run_config)
        (run_dir / "run_log.md").write_text(
            "\n".join(
                [
                    f"# {run_id}",
                    "",
                    f"- Owner: {OWNER}",
                    f"- Condition: {condition}",
                    f"- Seed: {seed}",
                    f"- CPT corpus files: {', '.join(cpt_files) if cpt_files else 'none'}",
                    "- Fine-tuning input: msa_train_v1.csv",
                    "- Validation input: msa_val_v1.csv",
                    "- Checkpoint rule: highest Macro-F1 on MSA validation",
                    f"- Best epoch: {best_epoch}",
                    f"- Macro-F1: {metrics['macro_f1']:.6f}",
                    "- Checkpoint reload: passed",
                    "- Saudi test accessed: false",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return summary
    except Exception:
        (run_dir / "failure_traceback.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        raise
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one clean SARF AraBERT condition seed or the CPT technical check."
    )
    parser.add_argument("--project-root", required=True)
    parser.add_argument(
        "--mode", choices=("official", "cpt_technical_check"), default="official"
    )
    parser.add_argument("--condition", choices=("E0", "E1", "E2", "E3", "EB"))
    parser.add_argument("--seed", type=int, choices=OFFICIAL_SEEDS)
    args = parser.parse_args()
    if args.mode == "cpt_technical_check":
        assert args.condition is None, "Do not pass --condition for the CPT technical check."
        assert args.seed is None, "Do not pass --seed for the CPT technical check."
        result = run_cpt_technical_check(args.project_root)
    else:
        assert args.condition is not None, "--condition is required for an official run."
        assert args.seed is not None, "--seed is required for an official run."
        result = run_official(args.project_root, args.condition, args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
