import argparse
import json
import os
import random
import sys
from collections import Counter

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
INDEX_PATH = os.path.join(
    PROJECT_ROOT, "app", "assets", "msa_train_retrieval_index.pt"
)
AUDIT_PATH = os.path.join(
    PROJECT_ROOT, "app", "assets", "msa_train_retrieval_index_audit.json"
)
MAPPING_PATH = os.path.join(PROJECT_ROOT, "arabert_label_mapping.json")

EXPECTED_ROWS = 10_732
EXPECTED_LABELS = 77
EXPECTED_DIM = 384

PROHIBITED_SOURCES = [
    "msa_val",
    "saudi_test",
    "saudi_frozen",
    "val_v1",
    "test_frozen",
]

passed = 0
failed = 0


def check(description, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {description}")
    else:
        failed += 1
        msg = f"  FAIL  {description}"
        if detail:
            msg += f" [{detail}]"
        print(msg)


def main():
    global passed, failed

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        default=None,
        help="Path to msa_train_v1.csv for random integrity check",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("MSA Retrieval Index Verification")
    print("=" * 60)

    print("\n[1] File existence")
    check("Index file exists", os.path.exists(INDEX_PATH))
    check("Audit file exists", os.path.exists(AUDIT_PATH))
    check("Label mapping exists", os.path.exists(MAPPING_PATH))

    if not os.path.exists(INDEX_PATH):
        print("\nCannot continue without index file.")
        sys.exit(1)

    index = torch.load(INDEX_PATH, map_location="cpu", weights_only=False)

    print("\n[2] Index structure")
    required_keys = ["texts", "labels", "embeddings", "source_file", "source_row_ids"]
    for key in required_keys:
        check(f"Key '{key}' present", key in index)

    texts = index.get("texts", [])
    labels = index.get("labels", [])
    embeddings = index.get("embeddings", torch.tensor([]))

    print("\n[3] Row counts")
    check(
        f"Text count = {EXPECTED_ROWS}",
        len(texts) == EXPECTED_ROWS,
        f"got {len(texts)}",
    )
    check(
        f"Label count = {EXPECTED_ROWS}",
        len(labels) == EXPECTED_ROWS,
        f"got {len(labels)}",
    )

    if isinstance(embeddings, torch.Tensor):
        emb_shape = tuple(embeddings.shape)
    else:
        emb_shape = embeddings.shape
    check(
        f"Embedding rows = {EXPECTED_ROWS}",
        emb_shape[0] == EXPECTED_ROWS,
        f"got {emb_shape[0]}",
    )

    source_row_ids = index.get("source_row_ids", [])
    check(
        f"Source row IDs count = {EXPECTED_ROWS}",
        len(source_row_ids) == EXPECTED_ROWS,
        f"got {len(source_row_ids)}",
    )
    check(
        "Source row IDs are unique",
        len(set(source_row_ids)) == len(source_row_ids),
        f"unique: {len(set(source_row_ids))}, total: {len(source_row_ids)}",
    )

    print("\n[4] Embedding properties")
    check(
        f"Embedding dim = {EXPECTED_DIM}",
        len(emb_shape) == 2 and emb_shape[1] == EXPECTED_DIM,
        f"got shape {emb_shape}",
    )
    check(
        "Embedding dtype = float16",
        embeddings.dtype == torch.float16,
        f"got {embeddings.dtype}",
    )

    print("\n[5] Label distribution")
    unique_labels = set(labels)
    check(
        f"Unique labels = {EXPECTED_LABELS}",
        len(unique_labels) == EXPECTED_LABELS,
        f"got {len(unique_labels)}",
    )

    label_counts = Counter(labels)
    all_present = all(v >= 1 for v in label_counts.values())
    check(
        "Every label has at least 1 example",
        all_present,
        f"empty labels: {[k for k, v in label_counts.items() if v < 1]}",
    )

    print("\n[6] Mapping compatibility")
    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    valid_labels = set(mapping["id_to_label"].values())
    unrecognized = unique_labels - valid_labels
    check(
        "All index labels in arabert_label_mapping.json",
        len(unrecognized) == 0,
        f"unrecognized: {unrecognized}",
    )

    print("\n[7] Data policy compliance")
    source = index.get("source_file", "")
    source_lower = source.lower()
    prohibited_match = any(p in source_lower for p in PROHIBITED_SOURCES)
    check(
        "Source file is not prohibited",
        not prohibited_match,
        f"source: {source}",
    )
    check(
        "Source file looks like MSA train",
        "train" in source_lower,
        f"source: {source}",
    )

    print("\n[8] Vector alignment spot-check")
    if isinstance(embeddings, torch.Tensor):
        emb_np = embeddings.numpy().astype(np.float32)
    else:
        emb_np = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(emb_np, axis=1)
    near_unit = np.allclose(norms, 1.0, atol=0.02)
    check(
        "Embeddings are approximately unit-normalized",
        near_unit,
        f"norm range: [{norms.min():.4f}, {norms.max():.4f}]",
    )

    no_nan = not np.any(np.isnan(emb_np))
    check("No NaN values in embeddings", no_nan)

    no_zero_rows = not np.any(norms < 0.01)
    check("No zero-vector rows", no_zero_rows)

    print("\n[9] Random integrity check against source CSV")
    if args.input_csv and os.path.exists(args.input_csv):
        df = pd.read_csv(args.input_csv)
        text_col = next(
            (c for c in ["text", "Text", "sentence", "Sentence"] if c in df.columns),
            df.columns[0],
        )
        label_col = next(
            (c for c in ["label", "Label", "intent", "Intent"] if c in df.columns),
            df.columns[1],
        )

        n_spot = min(50, len(source_row_ids))
        random.seed(2026)
        spot_indices = random.sample(range(len(source_row_ids)), n_spot)
        mismatches = 0
        for idx in spot_indices:
            row_id = source_row_ids[idx]
            csv_text = str(df.iloc[row_id][text_col])
            csv_label = str(df.iloc[row_id][label_col])
            if texts[idx] != csv_text or labels[idx] != csv_label:
                mismatches += 1
        check(
            f"Random spot-check ({n_spot} rows) matches source CSV",
            mismatches == 0,
            f"{mismatches} mismatches",
        )
    else:
        print("  SKIP  No --input-csv provided; skipping CSV integrity check")

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
