# baseline_rag.py
# A minimal starter RAG retrieval pipeline over corpus.jsonl.
# It runs, but its answer quality is poor. Your task is to diagnose why and improve it.
#
# Requirements (install locally):
#   pip install sentence-transformers numpy
# You may swap the embedding model for any local model you prefer.

import json
import numpy as np
from sentence_transformers import SentenceTransformer

CORPUS_PATH = "corpus.jsonl"
CHUNK_SIZE = 400
EMBED_MODEL = "all-MiniLM-L6-v2"


def load_docs(path):
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def chunk_text(text, size=CHUNK_SIZE):
    # fixed-size character windows
    return [text[i:i + size] for i in range(0, len(text), size)]


def build_index(docs, model):
    chunks = []
    for d in docs:
        for c in chunk_text(d["text"]):
            chunks.append({"doc_id": d["id"], "title": d["title"], "text": c})
    vectors = model.encode([c["text"] for c in chunks])
    vectors = np.asarray(vectors, dtype="float32")
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return chunks, vectors


def retrieve(query, chunks, vectors, model):
    q = model.encode([query])[0].astype("float32")
    q = q / np.linalg.norm(q)
    sims = vectors @ q
    best = int(np.argmax(sims))
    return chunks[best], float(sims[best])


def answer(query, chunks, vectors, model):
    hit, score = retrieve(query, chunks, vectors, model)
    # Returns the single best-matching chunk as the answer.
    return f"[{hit['doc_id']}] {hit['text']}"


if __name__ == "__main__":
    docs = load_docs(CORPUS_PATH)
    model = SentenceTransformer(EMBED_MODEL)
    chunks, vectors = build_index(docs, model)

    example_queries = [
        "What is the rated output of the C-100 compressor?",
        "How often should temperature sensors be calibrated?",
        "What should be checked before starting the M-50 motor?",
    ]
    for q in example_queries:
        print("Q:", q)
        print("A:", answer(q, chunks, vectors, model))
        print("-" * 60)
