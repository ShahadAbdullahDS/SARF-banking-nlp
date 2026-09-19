import json
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import torch
from arabert.preprocess import ArabertPreprocessor
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_PATH = os.path.join(BASE_DIR, "models", "best_checkpoint")
MAPPING_PATH = os.path.join(BASE_DIR, "arabert_label_mapping.json")
RETRIEVAL_INDEX_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "assets", "msa_train_retrieval_index.pt"
)
RETRIEVER_MODEL_PATH = os.path.join(BASE_DIR, "models", "minilm")

MODEL_NAME = "aubmindlab/bert-base-arabertv2"
MAX_LENGTH = 128

preprocessor = ArabertPreprocessor(model_name=MODEL_NAME, apply_farasa_segmentation=False)
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)
model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_PATH)
model.eval()

with open(MAPPING_PATH, "r", encoding="utf-8") as f:
    mapping = json.load(f)
id_to_label = mapping["id_to_label"]

_retrieval_index = None
_retriever_model = None


def _load_retrieval_index():
    global _retrieval_index
    if _retrieval_index is not None:
        return _retrieval_index
    if not os.path.exists(RETRIEVAL_INDEX_PATH):
        return None
    index = torch.load(RETRIEVAL_INDEX_PATH, map_location="cpu", weights_only=False)
    _retrieval_index = index
    return index


def _load_retriever():
    global _retriever_model
    if _retriever_model is not None:
        return _retriever_model
    if not os.path.isdir(RETRIEVER_MODEL_PATH):
        return None
    try:
        from sentence_transformers import SentenceTransformer
        _retriever_model = SentenceTransformer(RETRIEVER_MODEL_PATH)
        return _retriever_model
    except Exception:
        return None


def predict_unified(text: str):
    cleaned = preprocessor.preprocess(text)
    inputs = tokenizer(
        cleaned,
        return_tensors="pt",
        max_length=MAX_LENGTH,
        truncation=True,
        padding=True,
    )
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0]

    top3_vals = torch.topk(probs, k=3)
    top3 = []
    for score, idx in zip(top3_vals.values, top3_vals.indices):
        label = id_to_label[str(idx.item())]
        top3.append({"label": label, "score": round(score.item(), 4)})

    all_scores = [
        (id_to_label[str(i)], round(probs[i].item(), 4))
        for i in range(len(probs))
    ]

    return top3, all_scores


def predict(text: str):
    top3, _ = predict_unified(text)
    return top3


def predict_all(text: str):
    _, all_scores = predict_unified(text)
    return all_scores


def retrieve_similar_msa_examples(query: str, top_k: int = 3):
    index = _load_retrieval_index()
    if index is None:
        return []

    retriever = _load_retriever()
    if retriever is None:
        return []

    try:
        query_emb = retriever.encode([query], convert_to_numpy=True)
        query_vec = query_emb[0].astype(np.float32)
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-9)

        stored_embs = index["embeddings"]
        if isinstance(stored_embs, torch.Tensor):
            stored_embs = stored_embs.numpy()
        stored_embs = stored_embs.astype(np.float32)

        scores = stored_embs @ query_vec
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "text": index["texts"][idx],
                "label": index["labels"][idx],
                "score": round(float(scores[idx]), 4),
            })
        return results
    except Exception:
        return []
