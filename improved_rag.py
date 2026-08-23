import json
import os
import re
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from difflib import SequenceMatcher

import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# Cross-encoder is optional: if it can't be loaded (e.g. no local copy
# cached and no network), the pipeline degrades to dense+BM25 fusion
# only, instead of crashing. This matters for the offline constraint.
try:
    from sentence_transformers import CrossEncoder
    _HAS_CROSS_ENCODER = True
except ImportError:
    _HAS_CROSS_ENCODER = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CORPUS_PATH = os.environ.get("CORPUS_PATH", "./corpus.jsonl")

# These docs are short technical passages by design; the longest in this
# corpus is ~342 chars / 6 sentences. CHUNK_MAX_CHARS is the safety net
# for future, longer documents; CHUNK_MAX_SENTENCES is set generously so
# short docs stay intact as a single chunk instead of being split mid-answer.
CHUNK_MAX_SENTENCES = 8
CHUNK_MAX_CHARS = 150

DENSE_MODEL = os.environ.get("DENSE_MODEL", "all-MiniLM-L6-v2")
CROSS_ENCODER_MODEL = os.environ.get("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RRF_K = 60

# Single, consolidated confidence threshold.
# Calibrate with calibrate_threshold() against evaluation.jsonl.
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.25"))

STOPWORDS = {"the", "a", "an", "and", "or", "of", "for", "in", "on", "to", "is"}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    chunk_id: str        # unique id: "{doc_id}_chunk_{idx}"
    doc_id: str
    title: str
    text: str            # used for embedding/BM25 (metadata-prefixed if desired)
    raw_text: str        # plain text, used for display and keyword checks

    def __hash__(self):
        return hash(self.chunk_id)

    def __eq__(self, other):
        return isinstance(other, Chunk) and self.chunk_id == other.chunk_id


# ---------------------------------------------------------------------------
# 1. Data loading & basic cleaning
# ---------------------------------------------------------------------------
def load_docs(path: str) -> List[Dict]:
    """Load corpus with basic whitespace cleaning."""
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
            doc["text"] = re.sub(r"\s+", " ", doc.get("text", "")).strip()
            docs.append(doc)
    return docs


# ---------------------------------------------------------------------------
# 2. Sentence-aware chunking with metadata injection
# ---------------------------------------------------------------------------
def split_sentences(text: str) -> List[str]:
    """
    Regex-based sentence splitter.

    NOTE: we deliberately do NOT use nltk.sent_tokenize here. NLTK's
    punkt tokenizer triggers a network download the first time it's used
    on a machine without a cached copy, which silently violates an
    offline/on-prem constraint. A regex splitter is good enough for short,
    well-punctuated technical passages like this corpus.
    """
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text_sentences(
    text: str,
    title: str,
    doc_id: str,
    max_sentences: int = CHUNK_MAX_SENTENCES,
    max_chars: int = CHUNK_MAX_CHARS,
) -> List[Chunk]:
    """
    Sentence-aware chunking (replaces blind character windows, which
    routinely cut sentences — and therefore facts — in half).

    Each chunk holds up to `max_sentences` sentences and stays under
    `max_chars` characters. Title/doc_id metadata can be prepended to the
    embedding text so the embedding is anchored to which piece of equipment
    the chunk is about — this helps disambiguate DOC-01 vs DOC-02.
    """
    sentences = split_sentences(text)
    chunks: List[Chunk] = []
    current: List[str] = []
    current_len = 0
    idx = 0

    def flush():
        nonlocal idx
        raw = " ".join(current)
        # Metadata injection (uncomment if you want title/doc_id in embedding text):
        # embed_text = f"Title: {title}. Document: {doc_id}. {raw}"
        embed_text = raw
        chunks.append(Chunk(
            chunk_id=f"{doc_id}_chunk_{idx}",
            doc_id=doc_id,
            title=title,
            text=embed_text,
            raw_text=raw,
        ))
        idx += 1

    for sent in sentences:
        sent_len = len(sent)
        if current and (len(current) >= max_sentences or current_len + sent_len > max_chars):
            flush()
            current, current_len = [sent], sent_len
        else:
            current.append(sent)
            current_len += sent_len

    if current:
        flush()

    return chunks


# ---------------------------------------------------------------------------
# 3. Fuzzy matching (kept minimal)
# ---------------------------------------------------------------------------
def fuzzy_match_score(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


# ---------------------------------------------------------------------------
# 4. Hybrid index: dense + sparse, with RRF fusion and cross-encoder rerank
# ---------------------------------------------------------------------------
class HybridIndex:
    """
    Dense (embedding) + sparse (BM25) retrieval, fused with Reciprocal
    Rank Fusion, then re-ranked with a cross-encoder.

    Why hybrid: dense retrieval alone misses exact identifiers like
    "P-200" or "E-115" when their embedding neighborhood is crowded
    with other equipment codes; BM25 alone misses paraphrases like
    "how often should X be serviced" vs "preventive maintenance
    interval". Combining both and fusing with RRF (parameter-light,
    no training data required) is a good fit for a small, on-prem corpus.
    """

    def __init__(self, chunks: List[Chunk], dense_model: str = DENSE_MODEL,
                 cross_encoder_model: str = CROSS_ENCODER_MODEL):
        self.chunks = chunks
        self.chunk_texts = [c.text for c in chunks]

        print(f"[HybridIndex] Loading dense model: {dense_model}")
        self.dense_model = SentenceTransformer(dense_model)
        print(f"[HybridIndex] Encoding {len(chunks)} chunks...")
        vectors = self.dense_model.encode(self.chunk_texts, show_progress_bar=False)
        vectors = np.asarray(vectors, dtype="float32")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.dense_vectors = vectors / norms

        print("[HybridIndex] Building BM25 index...")
        tokenized = [self._tokenize(t) for t in self.chunk_texts]
        self.bm25 = BM25Okapi(tokenized)

        self.cross_encoder = None
        if _HAS_CROSS_ENCODER:
            try:
                print(f"[HybridIndex] Loading cross-encoder: {cross_encoder_model}")
                self.cross_encoder = CrossEncoder(cross_encoder_model)
            except Exception as e:
                print(f"[HybridIndex] WARNING: could not load cross-encoder ({e}); "
                      f"falling back to fusion-score ranking only.")

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenize for BM25, preserving hyphenated part numbers like P-200, E-115."""
        tokens = re.findall(r"[A-Z]*\d+[A-Z]*(?:-[A-Z]*\d+[A-Z]*)+|\b\w+\b", text.lower())
        return [t for t in tokens if len(t) > 1]

    def dense_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        q = self.dense_model.encode([query])[0].astype("float32")
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm
        sims = self.dense_vectors @ q
        top = np.argsort(sims)[::-1][:top_k]
        return [(int(i), float(sims[i])) for i in top]

    def sparse_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        scores = self.bm25.get_scores(self._tokenize(query))
        top = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top]

    def rrf_fusion(
        self,
        dense_results: List[Tuple[int, float]],
        sparse_results: List[Tuple[int, float]],
        k: int = RRF_K,
    ) -> List[Tuple[int, float]]:
        """score = sum(1 / (k + rank)) across both result lists."""
        scores: Dict[int, float] = {}
        for rank, (idx, _) in enumerate(dense_results, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
        for rank, (idx, _) in enumerate(sparse_results, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def rerank(self, query: str, chunk_indices: List[int]) -> List[Tuple[int, float]]:
        """Cross-encoder rerank; falls back to input order if unavailable."""
        if self.cross_encoder is None or not chunk_indices:
            return [(idx, 0.5) for idx in chunk_indices]
        pairs = [(query, self.chunk_texts[i]) for i in chunk_indices]
        scores = self.cross_encoder.predict(pairs, show_progress_bar=False)
        scored = list(zip(chunk_indices, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(idx, float(s)) for idx, s in scored]


# ---------------------------------------------------------------------------
# 5. Conflict / near-duplicate detection
# ---------------------------------------------------------------------------
def jaccard_similarity(text1: str, text2: str) -> float:
    set1 = set(re.findall(r"\b\w+\b", text1.lower()))
    set2 = set(re.findall(r"\b\w+\b", text2.lower()))
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def extract_numbers(text: str) -> List[str]:
    pattern = r"\d+(?:\.\d+)?\s*(?:bar|mm/s|kW|m3/h|m3/min|hours|°C|Celsius|kg|RPM|rpm)?"
    return [m.strip() for m in re.findall(pattern, text.lower()) if m.strip()]


def extract_entities(text: str) -> List[str]:
    """Equipment/error-code identifiers like P-200, C-100, E-115, BRG-4410."""
    return re.findall(r"\b[A-Z]{1,4}-\d+\b", text)


def detect_conflict(
    top_chunks: List[Chunk],
    near_dup_jaccard_threshold: float = 0.3,
) -> Optional[str]:
    """
    Flag two kinds of cross-document data-quality issues in the top-2
    results, rather than silently picking one value:

    1. Value conflict — both chunks discuss the same named equipment
       or error code (e.g. "P-200") but state different numeric
       values for the same kind of quantity (e.g. 16 bar vs 12 bar).
       Gated on a SHARED ENTITY, not on overall text similarity.
    2. Near-duplicate — both chunks share an entity, state the SAME
       numeric value, and have at least moderate text overlap.
       Informational only: the values agree, so this is corroborating
       evidence, not a discrepancy.
    """
    if len(top_chunks) < 2:
        return None
    c1, c2 = top_chunks[0], top_chunks[1]

    if c1.doc_id == c2.doc_id:
        return None

    entities1, entities2 = set(extract_entities(c1.raw_text)), set(extract_entities(c2.raw_text))
    shared_entities = entities1 & entities2
    if not shared_entities:
        return None  # different equipment entirely — nothing to compare

    nums1, nums2 = set(extract_numbers(c1.raw_text)), set(extract_numbers(c2.raw_text))
    if not nums1 or not nums2:
        return None

    jaccard = jaccard_similarity(c1.raw_text, c2.raw_text)
    entity_str = ", ".join(sorted(shared_entities))

    if nums1 == nums2 and jaccard > near_dup_jaccard_threshold:
        return (
            f"[NOTE: NEAR-DUPLICATE] {c1.chunk_id} and {c2.chunk_id} both describe "
            f"{entity_str} with matching values (Jaccard={jaccard:.2f}); treat as "
            f"corroborating, not conflicting."
        )

    if nums1 != nums2:
        return (
            f"[CONFLICT: VALUE MISMATCH] {c1.chunk_id} states {sorted(nums1)[:3]} for "
            f"{entity_str}; {c2.chunk_id} states {sorted(nums2)[:3]}. Please verify which is current."
        )
    return None


# ---------------------------------------------------------------------------
# 6. Abstention logic
# ---------------------------------------------------------------------------
def extract_attribute_from_query(query: str) -> Optional[str]:
    """
    Best-effort extraction of "the fact being asked about" from a
    question, used to check whether the retrieved chunk actually
    contains that fact (as opposed to just being topically related).
    """
    q = query.lower()
    patterns = [
        r"what is the ([\w\s]+?) (?:of|for|in)\b",
        r"who is the ([\w\s]+?) (?:of|for)\b",
        r"does .* have (?:an? )?([\w\s]+?)\?",
        r"what .*\b(weight|manufacturer|brand|warranty|purchase date|VFD|model|type|price|cost)\b",
    ]
    for pat in patterns:
        m = re.search(pat, q)
        if m:
            return m.group(1).strip()
    return None


def should_abstain(
    score: float,
    query: str,
    top_chunk: Optional[Chunk],
    threshold: float = CONFIDENCE_THRESHOLD,
) -> Tuple[bool, str]:
    """
    Single-threshold, two-check abstention decision:

      1. Score gate: if the top re-ranked score is below `threshold`,
         nothing relevant was found — abstain.
      2. Evidence gate: even with a topically-relevant chunk (score
         above threshold), check that the specific attribute or
         entity asked about actually appears in that chunk. A high
         similarity score only tells you the *topic* matches, not
         that the *fact* is present.
    """
    if top_chunk is None or score < threshold:
        return True, f"score {score:.3f} < threshold {threshold}"

    chunk_lower = top_chunk.raw_text.lower()

    attribute = extract_attribute_from_query(query)
    if attribute:
        attr_terms = [t for t in attribute.split() if t not in STOPWORDS]
        if attr_terms and not any(t in chunk_lower for t in attr_terms):
            return True, f"attribute '{attribute}' not found in {top_chunk.chunk_id}"

    for ent in extract_entities(query):
        if ent not in top_chunk.raw_text:
            return True, f"entity {ent} not found in {top_chunk.chunk_id}"

    return False, f"score {score:.3f} >= threshold, evidence present"


def ensure_diversity(
    reranked: List[Tuple[int, float]],
    chunks: List[Chunk],
    top_k: int = 3,
) -> List[Tuple[int, float]]:
    """Ensure top-k contains chunks from >=2 docs when possible."""
    if len(reranked) <= top_k:
        return reranked

    selected = reranked[:top_k]
    selected_doc_ids = {chunks[i].doc_id for i, _ in selected}

    if len(selected_doc_ids) >= 2:
        return reranked

    dominant_doc = chunks[selected[0][0]].doc_id
    for idx, score in reranked[top_k:]:
        if chunks[idx].doc_id != dominant_doc:
            new_selected = selected[:-1] + [(idx, score)]
            new_selected.sort(key=lambda x: x[1], reverse=True)
            return new_selected + reranked[top_k:]
    return reranked


# ---------------------------------------------------------------------------
# 7. Main retrieval pipeline
# ---------------------------------------------------------------------------
class ImprovedRAG:
    def __init__(
        self,
        corpus_path: str = CORPUS_PATH,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        dense_model: str = DENSE_MODEL,
        cross_encoder_model: str = CROSS_ENCODER_MODEL,
    ):
        self.confidence_threshold = confidence_threshold
        self.docs = load_docs(corpus_path)
        self.chunks: List[Chunk] = []
        for d in self.docs:
            self.chunks.extend(
                chunk_text_sentences(d["text"], d.get("title", ""), d["id"])
            )
        print(f"[ImprovedRAG] Loaded {len(self.docs)} docs -> {len(self.chunks)} chunks")
        self.index = HybridIndex(self.chunks, dense_model, cross_encoder_model)

    def retrieve(self, query: str, top_k: int = 3) -> Tuple[List[Dict[str, Any]], float, bool]:
        """
        Returns (top_k_chunks, best_score, abstained).

        IMPORTANT: `top_k_chunks` always reflects what was *actually
        retrieved*, regardless of the abstention decision. Abstention is
        reported purely via the `abstained` flag, so retrieval quality and
        abstention quality can be measured independently.
        """
        dense_results = self.index.dense_search(query, top_k=10)
        sparse_results = self.index.sparse_search(query, top_k=10)
        fused = self.index.rrf_fusion(dense_results, sparse_results)

        if not fused:
            return [], 0.0, True

        rerank_candidates = [idx for idx, _ in fused[:5]]
        reranked = self.index.rerank(query, rerank_candidates)
        reranked = ensure_diversity(reranked, self.chunks, top_k)

        top_chunks_obj: List[Chunk] = []
        top_scores: List[float] = []
        top_chunks_data: List[Dict[str, Any]] = []
        for idx, score in reranked[:top_k]:
            chunk = self.chunks[idx]
            top_chunks_obj.append(chunk)
            top_scores.append(float(score))
            top_chunks_data.append({
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "title": chunk.title,
                "text": chunk.raw_text,
                "rerank_score": float(score),
            })

        best_score = top_scores[0]
        best_chunk = top_chunks_obj[0]

        if best_score < self.confidence_threshold:
            return [], best_score, True
        
        conflict_msg = detect_conflict(top_chunks_obj)
        if conflict_msg:
            top_chunks_data[0]["conflict_note"] = conflict_msg

        abstained, reason = should_abstain(best_score, query, best_chunk, self.confidence_threshold)
        top_chunks_data[0]["abstention_reason"] = reason

        return top_chunks_data, best_score, abstained


# ---------------------------------------------------------------------------
# 8. Wrapper for evaluate.py
# ---------------------------------------------------------------------------
def make_improved_retriever(corpus_path: str = CORPUS_PATH,
                             confidence_threshold: float = CONFIDENCE_THRESHOLD):
    rag = ImprovedRAG(corpus_path, confidence_threshold=confidence_threshold)
    return lambda query, top_k=3: rag.retrieve(query, top_k)

def make_improved_retriever_v1(corpus_path: str = CORPUS_PATH,
                             confidence_threshold: float = CONFIDENCE_THRESHOLD):
    rag = ImprovedRAG(corpus_path, confidence_threshold=confidence_threshold)
    def _retrieve(query, top_k=3):
        chunks, score, abstained = rag.retrieve(query, top_k)
        if abstained:
            # User-facing: mask irrelevant chunks so callers don't act on them
            return "No relevant data found in corpus.", score, abstained
        return chunks, score, abstained
    return _retrieve
# ---------------------------------------------------------------------------
# 9. Threshold calibration (run once, results documented in README)
# ---------------------------------------------------------------------------
def calibrate_threshold(eval_path: str = "./evaluation.jsonl") -> None:
    rag = ImprovedRAG()
    questions = []
    with open(eval_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    print("\n--- Threshold Calibration ---")
    print(f"{'Type':13s} | {'Avg':>7s} | {'Min':>7s} | {'Max':>7s}")
    print("-" * 45)
    per_type_scores = {}
    for qtype in ["in_scope", "out_of_scope", "abstention"]:
        scores = []
        for q in questions:
            if q["type"] != qtype:
                continue
            _, score, _ = rag.retrieve(q["question"])
            scores.append(score)
        per_type_scores[qtype] = scores
        if scores:
            print(f"{qtype:13s} | {sum(scores)/len(scores):7.3f} | {min(scores):7.3f} | {max(scores):7.3f}")

    if per_type_scores.get("out_of_scope") and per_type_scores.get("in_scope"):
        suggested = (max(per_type_scores["out_of_scope"]) + min(per_type_scores["in_scope"])) / 2
        print(f"\nSuggested CONFIDENCE_THRESHOLD ~= {suggested:.3f} "
              f"(midpoint between max out-of-scope score and min in-scope score)")
    print("-" * 45)


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
        "What is the rated output of the C-100 compressor?",
        "How often should temperature sensors be calibrated?",
        "What should be checked before starting the M-50 motor?",
        "What is the acceptable vibration velocity limit for F-30 axial fan units, and how must the measurement be taken?",
    ]

    for q in test_queries:
        chunks, score, abstained = rag.retrieve(q)
        if score < 0.01:
            print(f"Q: {q}")
            print("  No relevant chunks found (score < 0.01)")
            print("-" * 60)
            continue
        print(f"Q: {q}")
        print(f"  Abstained: {abstained} | Score: {score:.4f}")
        for c in chunks:
            note = f" [{c['conflict_note']}]" if c.get("conflict_note") else ""
            print(f"  [{c['chunk_id']} / {c['doc_id']}]{note}")
            print(f"    {c['text'][:250]}")
        print("-" * 60)
        