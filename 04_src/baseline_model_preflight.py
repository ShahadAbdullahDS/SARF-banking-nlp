"""Inspect finalized SVM and TextCNN artifacts without opening the Saudi test.

This program is intentionally diagnostic only. It reads baseline checkpoints,
vocabulary, and label-mapping metadata, but it does not read any test file and
it does not compute model predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import torch

OWNER = "A"
NUM_LABELS = 77


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_value(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {
            "type": "dict",
            "key_count": len(value),
            "first_keys": [str(key) for key in list(value)[:10]],
        }
    if isinstance(value, list):
        return {"type": "list", "length": len(value), "first_values": value[:10]}
    return {"type": type(value).__name__, "repr": repr(value)[:500]}


def find_one(root: Path, alternatives: list[str], description: str) -> Path:
    candidates = [root / relative for relative in alternatives]
    found = [path for path in candidates if path.is_file()]
    assert len(found) == 1, (
        f"Expected exactly one {description}. Checked:\n" + "\n".join(str(path) for path in candidates)
    )
    return found[0]


def inspect(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root)
    baseline_root = root / "06_models_and_checkpoints" / "baselines"
    assert root.is_dir(), f"Project folder not found: {root}"
    assert baseline_root.is_dir(), f"Baseline model folder not found: {baseline_root}"

    svm_path = find_one(
        baseline_root,
        ["SVM/svm_tfidf.joblib", "svm/svm_tfidf.joblib"],
        "SVM joblib pipeline",
    )
    textcnn_path = find_one(
        baseline_root,
        ["textCnn/textcnn_best.pt", "textcnn/textcnn_best.pt", "TextCNN/textcnn_best.pt"],
        "TextCNN checkpoint",
    )
    vocab_path = find_one(
        baseline_root,
        ["textCnn/vocabulary.json", "textcnn/vocabulary.json", "TextCNN/vocabulary.json"],
        "TextCNN vocabulary",
    )
    mapping_path = find_one(
        baseline_root,
        ["textCnn/label_mapping.json", "textcnn/label_mapping.json", "TextCNN/label_mapping.json"],
        "TextCNN label mapping",
    )

    svm = joblib.load(svm_path)
    svm_summary: dict[str, Any] = {
        "path": str(svm_path),
        "python_type": f"{type(svm).__module__}.{type(svm).__name__}",
        "has_predict": callable(getattr(svm, "predict", None)),
    }
    if hasattr(svm, "named_steps"):
        svm_summary["named_steps"] = {
            name: f"{type(step).__module__}.{type(step).__name__}"
            for name, step in svm.named_steps.items()
        }
    if hasattr(svm, "classes_"):
        svm_summary["class_count"] = int(len(svm.classes_))
        svm_summary["classes_preview"] = [str(value) for value in svm.classes_[:10]]

    try:
        checkpoint = torch.load(textcnn_path, map_location="cpu", weights_only=True)
        checkpoint_load_status = "weights_only_loaded"
    except Exception as exc:
        checkpoint = None
        checkpoint_load_status = f"weights_only_load_failed: {type(exc).__name__}: {exc}"
    textcnn_summary: dict[str, Any] = {
        "path": str(textcnn_path),
        "weights_only_load_status": checkpoint_load_status,
    }
    if isinstance(checkpoint, dict):
        textcnn_summary["checkpoint_top_level"] = summarize_value(checkpoint)
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        if isinstance(state_dict, dict):
            textcnn_summary["state_dict_key_count"] = len(state_dict)
            textcnn_summary["state_dict_keys"] = [str(key) for key in list(state_dict)[:50]]
            textcnn_summary["tensor_shapes"] = {
                str(key): list(value.shape)
                for key, value in state_dict.items()
                if isinstance(value, torch.Tensor)
            }
        textcnn_summary["training_config"] = checkpoint.get("config")
        textcnn_summary["training_metadata"] = {
            str(key): summarize_value(value)
            for key, value in checkpoint.items()
            if key not in {"model_state_dict", "state_dict", "config"}
        }
    else:
        textcnn_summary["checkpoint_python_type"] = type(checkpoint).__name__ if checkpoint is not None else None

    vocab = read_json(vocab_path)
    mapping = read_json(mapping_path)
    mapping_label_to_id = mapping.get("label_to_id", {}) if isinstance(mapping, dict) else {}
    mapping_id_to_label = mapping.get("id_to_label", {}) if isinstance(mapping, dict) else {}
    assert len(mapping_label_to_id) == NUM_LABELS or len(mapping_id_to_label) == NUM_LABELS, (
        "TextCNN mapping must contain 77 labels."
    )
    payload = {
        "status": "baseline_preflight_passed_no_saudi_test_access",
        "owner": OWNER,
        "saudi_test_accessed": False,
        "svm": svm_summary,
        "textcnn": textcnn_summary,
        "vocabulary": {"path": str(vocab_path), **summarize_value(vocab)},
        "label_mapping": {
            "path": str(mapping_path),
            "label_to_id_count": len(mapping_label_to_id),
            "id_to_label_count": len(mapping_id_to_label),
            "top_level": summarize_value(mapping),
        },
        "next_step": "Use the reported checkpoint tensor shapes and metadata to construct a matching TextCNN evaluator; no Saudi test has been opened.",
    }
    output = root / "05_runs" / "baselines" / "baseline_preflight_no_saudi_test.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    args = parser.parse_args()
    result = inspect(args.project_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
