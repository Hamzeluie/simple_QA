"""
Evaluation Framework for Industrial RAG Retrieval
===================================================
Run with: python evaluate.py

Supports three question types:
  - "in_scope":      Answer exists in corpus → should retrieve and answer
  - "out_of_scope":  No relevant document → should abstain
  - "abstention":    Relevant document exists but fact is missing → should 
                     retrieve doc but abstain from answering

Metrics computed:
  - Hit@K:           Was the correct doc in the top-K retrieved chunks?
  - MRR:             Mean Reciprocal Rank of the first correct doc
  - Abstention Accuracy: % of questions that SHOULD abstain and DID abstain
  - False Abstention Rate: % of in-scope questions wrongly refused
  - Answer Coverage: % of expected keywords found in the returned text
  - Retrieval + Abstention Combined: For "abstention" type, did it both 
                                     find the doc AND abstain?

Design choices:
  - Hit@3 chosen because industrial queries are specific; the correct doc should
    appear in the top 3 chunks after re-ranking.
  - MRR chosen because it rewards ranking the correct doc higher.
  - Three-type design explicitly tests the hardest case: finding the right doc
    but admitting the answer is not there (hallucination control).
"""

import json
import numpy as np
from typing import List, Dict, Callable, Tuple, Any
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EVAL_PATH = "./evaluation.jsonl"
EVAL_PATH = "/home/mehdi/Documents/projects/knowledge_graph_examples/i4twins/evaluation.jsonl"
CORPUS_PATH = "./corpus.jsonl"
CORPUS_PATH = "/home/mehdi/Documents/projects/knowledge_graph_examples/i4twins/corpus.jsonl"
K = 3
ABSTENTION_MARKER = "no relevant data found"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class EvalResult:
    question: str
    question_type: str          # "in_scope" | "out_of_scope" | "abstention"
    category: str
    expected_doc_ids: List[str]
    expected_keywords: List[str]
    retrieved_doc_ids: List[str]
    retrieved_texts: List[str]
    score: float
    abstained: bool
    answer_text: str


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------
def compute_hit_at_k(result: EvalResult, k: int = K) -> int:
    """1 if any expected doc appears in top-k retrieved doc_ids, else 0.

    For out_of_scope: returns 0 (no expected docs).
    For abstention: still checks if the relevant doc was found.
    """
    if result.question_type == "out_of_scope":
        return 0
    retrieved_set = set(result.retrieved_doc_ids[:k])
    expected_set = set(result.expected_doc_ids)
    return 1 if retrieved_set & expected_set else 0


def compute_mrr(result: EvalResult) -> float:
    """Reciprocal rank of first correct doc in retrieved list."""
    if result.question_type == "out_of_scope":
        return 0.0
    for rank, doc_id in enumerate(result.retrieved_doc_ids, start=1):
        if doc_id in result.expected_doc_ids:
            return 1.0 / rank
    return 0.0


def compute_keyword_coverage(result: EvalResult) -> float:
    """Fraction of expected keywords present in the top-1 retrieved text.

    For abstention questions: if the system abstained, coverage is 1.0 
    (correct behavior). If it did NOT abstain, check if the text contains
    explicit "not mentioned" / "not specified" markers.
    """
    if not result.retrieved_texts:
        return 0.0

    text = result.retrieved_texts[0].lower()
    keywords = [kw.lower() for kw in result.expected_keywords]

    if result.question_type == "abstention":
        if result.abstained:
            return 1.0  # Correctly abstained
        # Did not abstain — check if it explicitly says info is missing
        missing_markers = ["not mentioned", "not specified", "not found", 
                          "no information", "not available"]
        if any(m in text for m in missing_markers):
            return 1.0
        # Otherwise, check if keywords are present
        matches = sum(1 for kw in keywords if kw in text)
        return matches / len(keywords) if keywords else 0.0

    # in_scope
    if result.abstained:
        return 0.0
    matches = sum(1 for kw in keywords if kw in text)
    return matches / len(keywords) if keywords else 1.0


def compute_abstention_correctness(result: EvalResult) -> int:
    """1 if abstention decision was correct, 0 otherwise.

    - out_of_scope: should abstain → 1 if abstained, 0 if not
    - abstention:   should abstain → 1 if abstained, 0 if not  
    - in_scope:     should NOT abstain → 1 if NOT abstained, 0 if abstained
    """
    should_abstain = result.question_type in ("out_of_scope", "abstention")
    if should_abstain:
        return 1 if result.abstained else 0
    else:
        return 1 if not result.abstained else 0


def compute_combined_abstention_retrieval(result: EvalResult, k: int = K) -> int:
    """For abstention-type questions only: 1 if BOTH doc was found AND 
    system abstained. This is the hardest metric."""
    if result.question_type != "abstention":
        return 0
    doc_found = compute_hit_at_k(result, k)
    abstained = 1 if result.abstained else 0
    return 1 if (doc_found and abstained) else 0


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------
def evaluate(
    retrieval_fn: Callable[[str], Tuple[List[Dict[str, Any]], float, bool]],
    eval_path: str = EVAL_PATH,
    k: int = K,
    abstention_marker: str = ABSTENTION_MARKER,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run evaluation over the question set.

    Args:
        retrieval_fn: callable(query) -> (top_k_chunks, best_score, abstained)
        eval_path: Path to evaluation.jsonl.
        k: Hit@K parameter.
        abstention_marker: String indicating abstention.
        verbose: Print per-question results.

    Returns:
        Dict with aggregated metrics and per-question details.
    """
    questions = []
    with open(eval_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    results: List[EvalResult] = []

    for q in questions:
        query = q["question"]
        q_type = q["type"]
        expected_doc_ids = q.get("expected_doc_ids", [])
        expected_keywords = q.get("expected_answer_contains", [])

        top_k_chunks, best_score, abstained = retrieval_fn(query)

        answer_text = abstention_marker if abstained else (
            top_k_chunks[0]["text"] if top_k_chunks else abstention_marker
        )
        if not abstained and answer_text.strip().lower() == abstention_marker.lower():
            abstained = True

        retrieved_doc_ids = [c["doc_id"] for c in top_k_chunks]
        retrieved_texts = [c["text"] for c in top_k_chunks]

        result = EvalResult(
            question=query,
            question_type=q_type,
            category=q.get("category", ""),
            expected_doc_ids=expected_doc_ids,
            expected_keywords=expected_keywords,
            retrieved_doc_ids=retrieved_doc_ids,
            retrieved_texts=retrieved_texts,
            score=best_score,
            abstained=abstained,
            answer_text=answer_text,
        )
        results.append(result)

        if verbose:
            print(f"Q: {query}")
            print(f"  Type: {q_type} | Category: {q['category']}")
            print(f"  Expected docs: {expected_doc_ids}")
            print(f"  Retrieved docs: {retrieved_doc_ids[:k]}")
            print(f"  Abstained: {abstained} | Score: {best_score:.4f}")
            print(f"  Answer: {answer_text[:150]}...")
            print("-" * 60)

    # -----------------------------------------------------------------------
    # Aggregate by type
    # -----------------------------------------------------------------------
    in_scope = [r for r in results if r.question_type == "in_scope"]
    out_scope = [r for r in results if r.question_type == "out_of_scope"]
    abstention = [r for r in results if r.question_type == "abstention"]
    all_should_abstain = out_scope + abstention

    # Retrieval metrics (in_scope + abstention)
    retrievable = in_scope + abstention
    hit_at_k_scores = [compute_hit_at_k(r, k) for r in retrievable]
    mrr_scores = [compute_mrr(r) for r in retrievable]
    hit_at_k = np.mean(hit_at_k_scores) if hit_at_k_scores else 0.0
    mrr = np.mean(mrr_scores) if mrr_scores else 0.0

    # Keyword coverage (in_scope only)
    kw_coverages = [compute_keyword_coverage(r) for r in in_scope]
    avg_kw_coverage = np.mean(kw_coverages) if kw_coverages else 0.0

    # Abstention metrics
    abstention_correct_all = [compute_abstention_correctness(r) for r in results]
    abstention_acc = np.mean(abstention_correct_all) if abstention_correct_all else 0.0

    false_abstention_in_scope = [1 if r.abstained else 0 for r in in_scope]
    false_abstention_rate = np.mean(false_abstention_in_scope) if false_abstention_in_scope else 0.0

    # Combined metric for abstention type (retrieval + abstention)
    combined_scores = [compute_combined_abstention_retrieval(r, k) for r in abstention]
    combined_rate = np.mean(combined_scores) if combined_scores else 0.0

    # Per-type breakdown
    in_scope_hit = np.mean([compute_hit_at_k(r, k) for r in in_scope]) if in_scope else 0.0
    abstention_hit = np.mean([compute_hit_at_k(r, k) for r in abstention]) if abstention else 0.0
    abstention_abstain_acc = np.mean([compute_abstention_correctness(r) for r in abstention]) if abstention else 0.0
    out_scope_abstain_acc = np.mean([compute_abstention_correctness(r) for r in out_scope]) if out_scope else 0.0

    metrics = {
        "hit_at_k": float(hit_at_k),
        "mrr": float(mrr),
        "avg_keyword_coverage": float(avg_kw_coverage),
        "abstention_accuracy": float(abstention_acc),
        "false_abstention_rate": float(false_abstention_rate),
        "combined_abstention_retrieval_rate": float(combined_rate),
        "in_scope_hit_at_k": float(in_scope_hit),
        "abstention_hit_at_k": float(abstention_hit),
        "abstention_type_abstain_accuracy": float(abstention_abstain_acc),
        "out_of_scope_abstain_accuracy": float(out_scope_abstain_acc),
        "k": k,
        "total_questions": len(questions),
        "in_scope_count": len(in_scope),
        "out_of_scope_count": len(out_scope),
        "abstention_count": len(abstention),
    }

    # -----------------------------------------------------------------------
    # Print summary
    # -----------------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 65)
        print("EVALUATION SUMMARY")
        print("=" * 65)
        print(f"Total questions: {len(questions)}  "
              f"(in_scope={len(in_scope)}, out_of_scope={len(out_scope)}, "
              f"abstention={len(abstention)})")
        print()
        print("Retrieval Quality (in_scope + abstention)")
        print(f"  Hit@{k}:    {hit_at_k:.3f}")
        print(f"  MRR:       {mrr:.3f}")
        print(f"  Keyword Coverage (in_scope only): {avg_kw_coverage:.3f}")
        print()
        print("Per-Type Breakdown")
        print(f"  In-scope Hit@{k}:        {in_scope_hit:.3f}")
        print(f"  Abstention Hit@{k}:      {abstention_hit:.3f}  "
              f"(found doc + abstained: {combined_rate:.3f})")
        print()
        print("Abstention Quality")
        print(f"  Overall Abstention Accuracy:   {abstention_acc:.3f}")
        print(f"  Out-of-scope Abstain Acc:      {out_scope_abstain_acc:.3f}")
        print(f"  Abstention-type Abstain Acc:   {abstention_abstain_acc:.3f}")
        print(f"  False Abstention Rate:         {false_abstention_rate:.3f}")
        print("=" * 65)

    return {
        "metrics": metrics,
        "per_question": [
            {
                "question": r.question,
                "type": r.question_type,
                "category": r.category,
                "expected_doc_ids": r.expected_doc_ids,
                "retrieved_doc_ids": r.retrieved_doc_ids[:k],
                "abstained": r.abstained,
                "score": r.score,
                "answer_text": r.answer_text,
                "hit_at_k": compute_hit_at_k(r, k),
                "mrr": compute_mrr(r),
                "abstention_correct": compute_abstention_correctness(r),
                "combined": compute_combined_abstention_retrieval(r, k),
                "keyword_coverage": compute_keyword_coverage(r),
            }
            for r in results
        ],
    }


# ---------------------------------------------------------------------------
# Baseline wrapper
# ---------------------------------------------------------------------------
def make_baseline_retriever(corpus_path: str = CORPUS_PATH):
    import json
    import numpy as np
    from sentence_transformers import SentenceTransformer

    docs = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))

    CHUNK_SIZE = 400
    def chunk_text(text, size=CHUNK_SIZE):
        return [text[i:i + size] for i in range(0, len(text), size)]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    chunks = []
    for d in docs:
        for c in chunk_text(d["text"]):
            chunks.append({"doc_id": d["id"], "title": d["title"], "text": c})
    vectors = model.encode([c["text"] for c in chunks])
    vectors = np.asarray(vectors, dtype="float32")
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def retrieve(query: str, top_k: int = 3):
        q = model.encode([query])[0].astype("float32")
        q = q / np.linalg.norm(q)
        sims = vectors @ q
        top_indices = np.argsort(sims)[::-1][:top_k]
        top_chunks = [chunks[i] for i in top_indices]
        best_score = float(sims[top_indices[0]]) if len(top_indices) > 0 else 0.0
        return top_chunks, best_score, False

    return retrieve


if __name__ == "__main__":
    print("Loading baseline retriever...")
    baseline_retrieve = make_baseline_retriever()

    print("\nRunning baseline evaluation...\n")
    baseline_report = evaluate(baseline_retrieve, verbose=True)

    with open("baseline_evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(baseline_report, f, indent=2, ensure_ascii=False)
    print("\nBaseline report saved to baseline_evaluation_report.json")