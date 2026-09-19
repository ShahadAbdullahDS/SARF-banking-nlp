import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

EXPECTED_TOTAL_ROWS = 10_732
EXPECTED_LABELS = 77
ENCODER_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MAPPING_PATH = os.path.join(PROJECT_ROOT, "arabert_label_mapping.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "app", "assets")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--encoder", default=ENCODER_NAME)
    args = parser.parse_args()

    assert os.path.basename(args.input_csv) == "msa_train_v1.csv", (
        "Retrieval index must be built only from msa_train_v1.csv"
    )

    if not os.path.exists(args.input_csv):
        print(f"ERROR: Input file not found: {args.input_csv}")
        sys.exit(1)

    source_sha256 = _sha256(args.input_csv)
    print(f"Source SHA-256: {source_sha256}")

    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    valid_labels = set(mapping["id_to_label"].values())

    print(f"Loading CSV: {args.input_csv}")
    df = pd.read_csv(args.input_csv)

    text_col = next(
        (c for c in ["text", "Text", "sentence", "Sentence"] if c in df.columns),
        df.columns[0],
    )
    label_col = next(
        (c for c in ["label", "Label", "intent", "Intent"] if c in df.columns),
        df.columns[1],
    )
    print(f"Using columns: text='{text_col}', label='{label_col}'")

    assert len(df) == EXPECTED_TOTAL_ROWS, (
        f"Expected {EXPECTED_TOTAL_ROWS} rows, got {len(df)}"
    )

    unique_labels = set(df[label_col].unique())
    assert len(unique_labels) == EXPECTED_LABELS, (
        f"Expected {EXPECTED_LABELS} unique labels, got {len(unique_labels)}"
    )

    unrecognized = unique_labels - valid_labels
    assert not unrecognized, (
        f"Labels not in arabert_label_mapping.json: {unrecognized}"
    )

    df = df.reset_index(names="source_row_id")

    texts = df[text_col].tolist()
    labels = df[label_col].tolist()
    source_row_ids = df["source_row_id"].astype(int).tolist()

    print(f"Indexing all {len(texts)} rows (no sampling)")

    print("Verifying text-label alignment against source rows...")
    for i, row_id in enumerate(source_row_ids):
        assert texts[i] == df.loc[df["source_row_id"] == row_id, text_col].values[0]
        assert labels[i] == df.loc[df["source_row_id"] == row_id, label_col].values[0]
    print("Alignment verified.")

    print(f"Loading encoder: {args.encoder}")
    encoder = SentenceTransformer(args.encoder)

    print("Encoding texts...")
    embeddings = encoder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-9)
    embeddings_f16 = embeddings.astype(np.float16)

    print(f"Embeddings shape: {embeddings_f16.shape}, dtype: {embeddings_f16.dtype}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    index_path = os.path.join(OUTPUT_DIR, "msa_train_retrieval_index.pt")
    index_data = {
        "texts": texts,
        "labels": labels,
        "embeddings": torch.tensor(embeddings_f16),
        "source_row_ids": source_row_ids,
        "source_file": os.path.basename(args.input_csv),
        "source_sha256": source_sha256,
        "encoder": args.encoder,
    }
    torch.save(index_data, index_path)
    print(f"Saved index to: {index_path}")

    audit_path = os.path.join(OUTPUT_DIR, "msa_train_retrieval_index_audit.json")
    label_counts = df[label_col].value_counts().to_dict()
    audit = {
        "source_file": os.path.basename(args.input_csv),
        "source_sha256": source_sha256,
        "source_rows": EXPECTED_TOTAL_ROWS,
        "indexed_rows": len(texts),
        "unique_labels": len(unique_labels),
        "encoder": args.encoder,
        "embedding_shape": list(embeddings_f16.shape),
        "embedding_dtype": str(embeddings_f16.dtype),
        "label_distribution": {k: int(v) for k, v in label_counts.items()},
        "source_row_ids_saved": True,
        "alignment_verified": True,
    }
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    print(f"Saved audit to: {audit_path}")
    print("Done.")


if __name__ == "__main__":
    main()
