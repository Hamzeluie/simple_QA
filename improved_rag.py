"""
Improved RAG Retrieval Pipeline for Industrial Documents
========================================================

Features:
- Sentence-aware chunking with metadata injection (title + doc_id)
- Hybrid retrieval: Dense (all-MiniLM-L6-v2) + Sparse (BM25)
- Reciprocal Rank Fusion (RRF) for combining dense and sparse scores
- Cross-encoder re-ranking (ms-marco-MiniLM-L-6-v2)
- Two-layer abstention logic:
    1. Score gate (cross-encoder threshold)
    2. Answer-presence gate (keyword/entity check)
- Conflict detection via Jaccard similarity between top results
- Fuzzy entity matching via difflib (offline, no extra deps)

All models are local and offline-compatible.
"""

import json
import re
import os
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import numpy as np

# NLTK for sentence tokenization
try:
    from nltk.tokenize import sent_tokenize
except ImportError:
    import nltk
    nltk.download("punkt", quiet=True)
    from nltk.tokenize import sent_tokenize

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# Cross-encoder may not be available in all environments; handle gracefully
try:
    from sentence_transformers import CrossEncoder
    _HAS_CROSS_ENCODER = True
except ImportError:
    _HAS_CROSS_ENCODER = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CORPUS_PATH = "./corpus.jsonl"
CORPUS_PATH = "/home/mehdi/Documents/projects/knowledge_graph_examples/i4twins/corpus.jsonl"
CHUNK_MAX_SENTENCES = 30
CHUNK_MAX_CHARS = 1000
DENSE_MODEL = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RRF_K = 60

# Abstention thresholds (calibrated via evaluation set)
ABSTENTION_THRESHOLD_LOW = 0.25   # Below this: definitely abstain
ABSTENTION_THRESHOLD_HIGH = 0.65  # Above this: definitely answer


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    doc_id: str
    title: str
    text: str          # with metadata prefix for embedding
    raw_text: str      # without metadata prefix for display

    def __hash__(self):
        return hash((self.doc_id, self.raw_text))

    def __eq__(self, other):
        return self.doc_id == other.doc_id and self.raw_text == other.raw_text


# ---------------------------------------------------------------------------
# 1. Data loading & basic cleaning
# ---------------------------------------------------------------------------
def load_docs(path: str) -> List[Dict]:
    """Load corpus with basic cleaning."""
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Basic cleaning
            text = doc.get("text", "")
            # Normalize whitespace
            text = re.sub(r"\s+", " ", text).strip()
            doc["text"] = text
            docs.append(doc)
    return docs


# ---------------------------------------------------------------------------
# 2. Sentence-aware chunking with metadata injection
# ---------------------------------------------------------------------------
def split_sentences(text: str, use_nltk: bool = True) -> List[str]:
    """Split text into sentences. Falls back to simple regex if NLTK fails."""
    
    try:
        if use_nltk:
            return sent_tokenize(text)
        
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])|\.(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]
    except Exception:
        # Fallback regex-based sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])|\.(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]


def chunk_text_sentences(
    text: str,
    title: str,
    doc_id: str,
    max_sentences: int = CHUNK_MAX_SENTENCES,
    max_chars: int = CHUNK_MAX_CHARS,
) -> List[Chunk]:
    """
    Split text into sentence-aware chunks.

    Each chunk:
    - Contains up to `max_sentences` sentences
    - Does not exceed `max_chars` characters
    - Has metadata (title, doc_id) prepended for embedding

    Metadata injection rationale:
    - The title anchors the chunk to its document topic.
    - The doc_id provides disambiguation when multiple docs mention
      the same entity (e.g., P-200 in DOC-01 vs DOC-02).
    """
    sentences = split_sentences(text, False)
    chunks: List[Chunk] = []
    current_sentences: List[str] = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent)
        if (len(current_sentences) >= max_sentences or
                (current_len + sent_len > max_chars and current_sentences)):
            raw_text = " ".join(current_sentences)
            metadata = f"Title: {title}. Document: {doc_id}. "
            chunks.append(Chunk(
                doc_id=doc_id,
                title=title,
                text=metadata + raw_text,
                raw_text=raw_text,
            ))
            current_sentences = [sent]
            current_len = sent_len
        else:
            current_sentences.append(sent)
            current_len += sent_len

    # Flush remaining sentences
    if current_sentences:
        raw_text = " ".join(current_sentences)
        metadata = f"Title: {title}. Document: {doc_id}. "
        chunks.append(Chunk(
            doc_id=doc_id,
            title=title,
            text=metadata + raw_text,
            raw_text=raw_text,
        ))

    return chunks


# ---------------------------------------------------------------------------
# 3. Fuzzy entity matching (Levenshtein-like via difflib)
# ---------------------------------------------------------------------------
def fuzzy_match_score(a: str, b: str) -> float:
    """Normalized similarity ratio [0,1] using SequenceMatcher."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def find_best_fuzzy_match(query_term: str, candidates: List[str], threshold: float = 0.75) -> Optional[str]:
    """Find the candidate with highest fuzzy similarity above threshold."""
    best_score = 0.0
    best_match = None
    for cand in candidates:
        score = fuzzy_match_score(query_term, cand)
        if score > best_score and score >= threshold:
            best_score = score
            best_match = cand
    return best_match


# ---------------------------------------------------------------------------
# 4. Hybrid Index: Dense + Sparse
# ---------------------------------------------------------------------------
class HybridIndex:
    """
    Manages dense (embedding) and sparse (BM25) indices.
    Provides hybrid search with RRF fusion and cross-encoder re-ranking.
    """

    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.chunk_texts = [c.text for c in chunks]
        self.chunk_raw_texts = [c.raw_text for c in chunks]

        # Dense index
        print(f"[HybridIndex] Loading dense model: {DENSE_MODEL}")
        self.dense_model = SentenceTransformer(DENSE_MODEL)
        print(f"[HybridIndex] Encoding {len(chunks)} chunks...")
        self.dense_vectors = self.dense_model.encode(
            self.chunk_texts, show_progress_bar=False
        )
        self.dense_vectors = np.asarray(self.dense_vectors, dtype="float32")
        norms = np.linalg.norm(self.dense_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # avoid div-by-zero
        self.dense_vectors = self.dense_vectors / norms

        # Sparse index (BM25)
        print("[HybridIndex] Building BM25 index...")
        tokenized = [self._tokenize(t) for t in self.chunk_texts]
        self.bm25 = BM25Okapi(tokenized)

        # Cross-encoder for re-ranking
        if _HAS_CROSS_ENCODER:
            print(f"[HybridIndex] Loading cross-encoder: {CROSS_ENCODER_MODEL}")
            self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
        else:
            print("[HybridIndex] WARNING: CrossEncoder not available, skipping re-rank")
            self.cross_encoder = None

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace + punctuation tokenization for BM25."""
        return re.findall(r"\b\w+\b", text.lower())

    # -----------------------------------------------------------------------
    # Search methods
    # -----------------------------------------------------------------------
    def dense_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Return (chunk_idx, cosine_similarity) sorted descending."""
        q_vec = self.dense_model.encode([query])[0].astype("float32")
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec = q_vec / norm
        sims = self.dense_vectors @ q_vec
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [(int(i), float(sims[i])) for i in top_indices]

    def sparse_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Return (chunk_idx, bm25_score) sorted descending."""
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices]

    def rrf_fusion(
        self,
        dense_results: List[Tuple[int, float]],
        sparse_results: List[Tuple[int, float]],
        k: int = RRF_K,
    ) -> List[Tuple[int, float]]:
        """
        Reciprocal Rank Fusion.

        score = Σ 1 / (k + rank)  for each result list.

        Rationale: parameter-light, no training needed, works well for
        small corpora where learned fusion weights would overfit.
        """
        scores: Dict[int, float] = {}
        for rank, (idx, _) in enumerate(dense_results, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
        for rank, (idx, _) in enumerate(sparse_results, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return fused

    def rerank(
        self, query: str, chunk_indices: List[int]
    ) -> List[Tuple[int, float]]:
        """Re-rank candidate chunks using cross-encoder."""
        if self.cross_encoder is None or not chunk_indices:
            # Fallback: return in input order with dummy scores
            return [(idx, 0.5) for idx in chunk_indices]

        pairs = [(query, self.chunk_texts[i]) for i in chunk_indices]
        scores = self.cross_encoder.predict(pairs, show_progress_bar=False)
        scored = list(zip(chunk_indices, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


# ---------------------------------------------------------------------------
# 5. Conflict detection
# ---------------------------------------------------------------------------
def jaccard_similarity(text1: str, text2: str) -> float:
    """Compute Jaccard similarity over word sets."""
    set1 = set(re.findall(r"\b\w+\b", text1.lower()))
    set2 = set(re.findall(r"\b\w+\b", text2.lower()))
    if not set1 or not set2:
        return 0.0
    inter = set1 & set2
    union = set1 | set2
    return len(inter) / len(union)


def extract_numbers(text: str) -> List[str]:
    """Extract numeric values (with units) from text."""
    # Match patterns like "16 bar", "4.5 mm/s", "4000 hours"
    pattern = r"\d+(?:\.\d+)?\s*(?:bar|mm/s|kW|m3/h|hours|°C|Celsius|kg|RPM|rpm)?"
    return re.findall(pattern, text.lower())


def detect_conflict(
    top_chunks: List[Chunk],
    top_scores: List[float],
    jaccard_threshold: float = 0.65,
    score_threshold: float = 0.3,
) -> Optional[str]:
    """
    Detect potential conflicts in top-2 results from different documents.

    Two signals of conflict:
    1. Near-duplicate content (high Jaccard) from different docs → 
       likely outdated copies or revisions.
    2. Same entity mentioned with different numeric values → 
       explicit specification conflict.

    Returns conflict message or None.
    """
    if len(top_chunks) < 2:
        return None

    c1, c2 = top_chunks[0], top_chunks[1]
    if c1.doc_id == c2.doc_id:
        return None

    # Both results must be reasonably confident
    if top_scores[1] < score_threshold:
        return None

    jaccard = jaccard_similarity(c1.raw_text, c2.raw_text)
    nums1 = set(extract_numbers(c1.raw_text))
    nums2 = set(extract_numbers(c2.raw_text))

    # Signal 1: Near-duplicate from different docs
    if jaccard > jaccard_threshold:
        return (
            f"[CONFLICT: NEAR-DUPLICATE] Doc-{c1.doc_id} and Doc-{c2.doc_id} "
            f"contain highly similar content (Jaccard={jaccard:.2f}). "
            f"One may be an outdated revision."
        )

    # Signal 2: Different numeric values for same topic
    if nums1 and nums2 and nums1 != nums2 and jaccard > 0.35:
        diff_nums = nums1.symmetric_difference(nums2)
        if diff_nums:
            return (
                f"[CONFLICT: VALUE MISMATCH] Doc-{c1.doc_id} states "
                f"{', '.join(sorted(nums1)[:3])}; Doc-{c2.doc_id} states "
                f"{', '.join(sorted(nums2)[:3])}. "
                f"Please verify which is current."
            )

    return None


# ---------------------------------------------------------------------------
# 6. Abstention logic
# ---------------------------------------------------------------------------
def extract_attribute_from_query(query: str) -> Optional[str]:
    """
    Extract the attribute being asked about.

    Examples:
    - "What is the weight of..." -> "weight"
    - "Who is the manufacturer..." -> "manufacturer"
    - "Does the ... have a VFD?" -> "VFD"
    """
    query_lower = query.lower()

    # Pattern: "What is the X of Y?" or "What is the X for Y?"
    m = re.search(r"what is the ([\w\s]+?) (?:of|for)", query_lower)
    if m:
        return m.group(1).strip()

    # Pattern: "Who is the X of Y?"
    m = re.search(r"who is the ([\w\s]+?) (?:of|for)", query_lower)
    if m:
        return m.group(1).strip()

    # Pattern: "Does ... have a/an X?"
    m = re.search(r"does .* have (?:an? )?([\w\s]+?)\?", query_lower)
    if m:
        return m.group(1).strip()

    # Pattern: "What ... X ...?" (broader catch)
    m = re.search(r"what .*\b(weight|manufacturer|brand|warranty|purchase date|VFD|model|type)\b", query_lower)
    if m:
        return m.group(1).strip()

    return None


def should_abstain(
    reranked_score: float,
    query: str,
    top_chunk: Chunk,
    threshold_low: float = ABSTENTION_THRESHOLD_LOW,
    threshold_high: float = ABSTENTION_THRESHOLD_HIGH,
) -> Tuple[bool, str]:
    """
    Two-layer abstention logic.

    Layer 1 — Score Gate:
        - score < threshold_low  → abstain (low confidence)
        - score > threshold_high → answer (high confidence)
        - in between → proceed to Layer 2

    Layer 2 — Answer Presence Gate:
        - Extract the attribute being asked about.
        - Check if the chunk text contains evidence of that attribute.
        - If the text mentions the equipment but not the attribute → abstain.

    Returns: (abstained, reason)
    """
    # Layer 1: Score gate
    if reranked_score < threshold_low:
        return True, f"re-ranker score {reranked_score:.3f} < {threshold_low}"

    if reranked_score >= threshold_high:
        return False, "high confidence"

    # Layer 2: Answer presence gate (uncertainty zone)
    attribute = extract_attribute_from_query(query)
    if attribute:
        chunk_lower = top_chunk.raw_text.lower()
        # Check if attribute or synonyms appear in chunk
        attr_terms = attribute.split()
        # Also check for common value indicators near the attribute
        found = any(term in chunk_lower for term in attr_terms)

        # Special cases: if asking for a number/brand and chunk has no numbers
        # or brand-like words near the entity, likely missing
        if not found:
            return (
                True,
                f"attribute '{attribute}' not found in retrieved text "
                f"(score {reranked_score:.3f} in uncertainty zone)",
            )

    return False, f"score {reranked_score:.3f} in uncertainty zone but attribute found"


# ---------------------------------------------------------------------------
# 7. Main retrieval pipeline
# ---------------------------------------------------------------------------
class ImprovedRAG:
    """
    End-to-end improved retrieval pipeline.
    """

    def __init__(self, corpus_path: str = CORPUS_PATH):
        self.docs = load_docs(corpus_path)
        self.chunks: List[Chunk] = []
        for d in self.docs:
            self.chunks.extend(chunk_text_sentences(d["text"], d.get("title", ""), d["id"]))
        print(f"[ImprovedRAG] Loaded {len(self.docs)} docs → {len(self.chunks)} chunks")
        self.index = HybridIndex(self.chunks)

    def retrieve(
        self, query: str, top_k: int = 3
    ) -> Tuple[List[Dict[str, Any]], float, bool]:
        """
        Retrieve top-k chunks for a query.

        Returns:
            top_k_chunks: list of dicts with keys doc_id, title, text
            best_score:   float, re-ranker score of top result
            abstained:    bool
        """
        # Step 1: Hybrid retrieval
        dense_results = self.index.dense_search(query, top_k=10)
        sparse_results = self.index.sparse_search(query, top_k=10)
        fused = self.index.rrf_fusion(dense_results, sparse_results)

        # Step 2: Re-rank top 5
        rerank_candidates = [idx for idx, _ in fused[:5]]
        reranked = self.index.rerank(query, rerank_candidates)

        if not reranked:
            return [], 0.0, True

        # Step 3: Build top-k output
        top_chunks_data: List[Dict[str, Any]] = []
        top_chunks_obj: List[Chunk] = []
        top_scores: List[float] = []

        for idx, score in reranked[:top_k]:
            chunk = self.chunks[idx]
            top_chunks_obj.append(chunk)
            top_scores.append(float(score))
            top_chunks_data.append({
                "doc_id": chunk.doc_id,
                "title": chunk.title,
                "text": chunk.raw_text,
                "rerank_score": float(score),
            })

        best_score = top_scores[0]
        best_chunk = top_chunks_obj[0]

        # Step 4: Conflict detection
        conflict_msg = detect_conflict(top_chunks_obj, top_scores)
        if conflict_msg:
            # Prepend conflict warning to the first chunk's text
            top_chunks_data[0]["text"] = conflict_msg + "\n" + top_chunks_data[0]["text"]
            top_chunks_data[0]["conflict"] = True

        # Step 5: Abstention check
        abstained, reason = should_abstain(best_score, query, best_chunk)
        if abstained:
            return (
                [{
                    "doc_id": "N/A",
                    "title": "N/A",
                    "text": "no relevant data found",
                    "rerank_score": best_score,
                    "abstention_reason": reason,
                }],
                best_score,
                True,
            )

        return top_chunks_data, best_score, False


# ---------------------------------------------------------------------------
# 8. Wrapper for evaluate.py
# ---------------------------------------------------------------------------
def make_improved_retriever(corpus_path: str = CORPUS_PATH):
    """Factory function matching evaluate.py interface."""
    rag = ImprovedRAG(corpus_path)
    def retrieve(query: str, top_k: int = 3):
        return rag.retrieve(query, top_k)
    return retrieve


# ---------------------------------------------------------------------------
# 9. Threshold calibration (optional, run once)
# ---------------------------------------------------------------------------
def calibrate_threshold(eval_path: str = "evaluation.jsonl") -> None:
    """
    Calibrate abstention thresholds using the evaluation set.
    Prints suggested thresholds based on score distributions.
    """
    import json

    rag = ImprovedRAG()
    questions = []
    with open(eval_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    print("\n--- Threshold Calibration ---")
    print("Type          | Avg Score | Min Score | Max Score")
    print("-" * 55)

    for qtype in ["in_scope", "out_of_scope", "abstention"]:
        scores = []
        for q in questions:
            if q["type"] != qtype:
                continue
            _, score, _ = rag.retrieve(q["question"])
            scores.append(score)
        if scores:
            print(
                f"{qtype:13s} | {sum(scores)/len(scores):9.3f} | "
                f"{min(scores):9.3f} | {max(scores):9.3f}"
            )

    print("\nSuggested thresholds:")
    print("  threshold_low  ≈ max(out_of_scope scores) + 0.05")
    print("  threshold_high ≈ min(in_scope scores) - 0.05")
    print("-" * 55)


# ---------------------------------------------------------------------------
# 10. CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--calibrate":
        calibrate_threshold()
        sys.exit(0)

    rag = ImprovedRAG()

    test_queries = [
        "What is the nominal flow rate and operating temperature range of the P-200 centrifugal process pump?",
        "What is the recommended oil change interval for the Turbine T-99?",
        "Who is the manufacturer of the P-200 centrifugal process pump?",
        "There is a discrepancy in the documentation regarding the P-200 pump's maximum operating pressure. What are the two different values stated?",
        "What is the Wi-Fi password for the compressor room?",
    ]

    for q in test_queries:
        print(f"Q: {q}")
        chunks, score, abstained = rag.retrieve(q)
        print(f"  Abstained: {abstained} | Score: {score:.4f}")
        for c in chunks:
            flag = " [CONFLICT]" if c.get("conflict") else ""
            print(f"  [{c['doc_id']}]{flag} {c['text'][:200]}...")
        print("-" * 60)