import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from predictor import predict, retrieve_similar_msa_examples

test_messages = [
    "بطاقتي ضاعت وأبغى أوقفها",
    "أبغى أحول فلوس لحساب ثاني",
    "ما قدرت أسحب من الصراف",
]

print("=" * 50)
print("AraBERT Smoke Test")
print("=" * 50)

all_passed = True

for msg in test_messages:
    print(f"\nInput: {msg}")

    results = predict(msg)
    assert len(results) == 3, f"Expected 3 results, got {len(results)}"
    assert all("label" in r and "score" in r for r in results), "Missing keys"
    assert results[0]["score"] >= results[1]["score"], "Scores not sorted"

    print("Top-3 Intents:")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['label']}  {r['score']}")

    retrieved = retrieve_similar_msa_examples(msg, top_k=3)

    assert retrieved, (
        "Semantic retrieval is unavailable. Check the local MiniLM model "
        "and msa_train_retrieval_index.pt."
    )
    assert 1 <= len(retrieved) <= 3
    assert all(
        isinstance(r["text"], str) and r["text"].strip()
        and isinstance(r["label"], str) and r["label"].strip()
        and isinstance(r["score"], float)
        and -1.0 <= r["score"] <= 1.0
        for r in retrieved
    )

    print("Similar MSA training examples:")
    for i, r in enumerate(retrieved, 1):
        print(f"  {i}. [{r['label']}] {r['text']}  sim={r['score']}")

    distinct = len(set(r["label"] for r in retrieved))
    print(f"  Distinct intents in top-3: {distinct}")

print("\n" + "=" * 50)
print("Smoke test passed")
