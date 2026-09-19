import argparse
import json
import os
import sys

import pandas as pd
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MAPPING_PATH = os.path.join(PROJECT_ROOT, "arabert_label_mapping.json")
INDEX_PATH = os.path.join(
    PROJECT_ROOT, "app", "assets", "msa_train_retrieval_index.pt"
)

SUSPECT_TEXTS = [
    "كيف أعرف الرقم السري الخاص بي؟",
    "لا أرى الرقم السري للبطاقة في أي مكان؟",
]
DISPLAYED_LABEL = "الحصول على بطاقة فعلية"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    args = parser.parse_args()

    print("=" * 60)
    print("PIN Label Investigation")
    print("=" * 60)

    df = pd.read_csv(args.input_csv)
    text_col = next(
        (c for c in ["text", "Text", "sentence", "Sentence"] if c in df.columns),
        df.columns[0],
    )
    label_col = next(
        (c for c in ["label", "Label", "intent", "Intent"] if c in df.columns),
        df.columns[1],
    )

    print(f"\n[1] Check suspect texts in source CSV ({args.input_csv})")
    for txt in SUSPECT_TEXTS:
        matches = df[df[text_col] == txt]
        if matches.empty:
            print(f"\n  Text: {txt}")
            print(f"  NOT FOUND in CSV")
        else:
            for _, row in matches.iterrows():
                print(f"\n  Text: {txt}")
                print(f"  CSV label: {row[label_col]}")
                print(f"  Displayed as: {DISPLAYED_LABEL}")
                if row[label_col] == DISPLAYED_LABEL:
                    print(f"  VERDICT: Label comes from the CSV (not an alignment error)")
                else:
                    print(f"  VERDICT: MISMATCH - CSV says '{row[label_col]}' but displayed as '{DISPLAYED_LABEL}'")

    print(f"\n[2] Check if texts exist in retrieval index")
    if os.path.exists(INDEX_PATH):
        index = torch.load(INDEX_PATH, map_location="cpu", weights_only=False)
        idx_texts = index.get("texts", [])
        idx_labels = index.get("labels", [])
        for txt in SUSPECT_TEXTS:
            found = [i for i, t in enumerate(idx_texts) if t == txt]
            if found:
                for i in found:
                    print(f"\n  Text: {txt}")
                    print(f"  Index position: {i}")
                    print(f"  Index label: {idx_labels[i]}")
            else:
                print(f"\n  Text: {txt}")
                print(f"  NOT in retrieval index")
    else:
        print("  Index file not found; skipping")

    print(f"\n[3] All CSV rows labeled '{DISPLAYED_LABEL}' containing 'السري'")
    mask = (df[label_col] == DISPLAYED_LABEL) & (df[text_col].str.contains("السري", na=False))
    subset = df[mask]
    if subset.empty:
        print(f"  No rows with label '{DISPLAYED_LABEL}' contain 'السري'")
    else:
        for _, row in subset.iterrows():
            print(f"  [{row[label_col]}] {row[text_col]}")

    print(f"\n[4] PIN-related labels in CSV")
    pin_labels = [l for l in df[label_col].unique() if "السري" in l or "رمز" in l or "PIN" in l.upper()]
    for lbl in sorted(pin_labels):
        count = len(df[df[label_col] == lbl])
        print(f"  {lbl} ({count} rows)")
        sample = df[df[label_col] == lbl].head(3)
        for _, row in sample.iterrows():
            print(f"    -> {row[text_col]}")

    print("\n" + "=" * 60)
    print("Investigation complete")


if __name__ == "__main__":
    main()
