# simple_QA

## Data Quality Issues
### 1. Near-Duplicates
**DOC-05** ("Fan F-30 — Vibration Limits") and **DOC-06** ("Vibration Note — F-30 Units") are near-duplicates. Both state the exact same acceptable vibration velocity limit (4.5 mm/s RMS at the bearing), the same potential causes for exceeding it (imbalance or worn bearing), and the same measurement requirement (three axes).
### 2. Conflicting Facts
Maximum Operating Pressure for Pump P-200:
**DOC-01** states: "The maximum operating pressure is 16 bar."
**DOC-02** states: "Do not exceed the rated pressure; for this unit the maximum operating pressure is 12 bar."
(This is a direct factual conflict that would need resolution in a real-world knowledge base).

## Evaluation Framework

### Design Philosophy

The evaluation framework was built **before** any retrieval improvements to serve as an unbiased compass. Without a reproducible benchmark, "improvements" are just intuition. This framework measures exactly what the task requires: retrieval quality and abstention correctness.

### Metrics Chosen

| Metric | Why it was chosen |
|--------|-------------------|
| **Hit@3** | Industrial queries are specific (part numbers, calibration intervals). The correct document should appear in the top 3 retrieved chunks. Hit@1 is too strict for chunked documents; Hit@5 is too lenient for a 16-document corpus. |
| **MRR** | Rewards ranking the correct document higher. If the true answer is at rank 3, MRR = 0.33; if at rank 1, MRR = 1.0. This discourages "correct but buried" retrieval. |
| **Keyword Coverage** | Fraction of expected keywords found in the top-1 retrieved text. Ensures the retrieved chunk actually contains the answer, not just the right document title. |
| **Abstention Accuracy** | % of out-of-scope questions correctly refused. **Critical requirement** from the task. A system that never abstains scores 0 here. |
| **False Abstention Rate** | % of in-scope questions wrongly refused. Measures if the abstention threshold is too aggressive. |

### Evaluation Dataset (`evaluation.jsonl`)

**12 questions total:**
- **8 in-scope**: Questions answerable from the corpus. Each has `expected_doc_ids` and `expected_answer_contains` (keywords that must appear in the answer text).
- **4 out-of-scope**: Questions about entities not in the corpus (e.g., "X-9000 turbine", "Z-200 pump") or non-technical topics (e.g., "CEO of the company").

**Categories covered:**
- `specific_fact_extraction`: Part numbers, ratings, pressures, power consumption.
- `frequency_lookup`: Calibration intervals, maintenance schedules.
- `procedure_lookup`: Pre-start checks, maintenance steps.
- `abstention_unknown_entity`: Equipment not mentioned in any document.
- `abstention_non_technical`: Questions outside the technical domain.

### How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run baseline evaluation
python evaluate.py

# 3. (After implementing improved pipeline) Run improved evaluation
python evaluate_improved.py
```

### Expected Output

The script prints per-question results and a summary table:

```
============================================================
EVALUATION SUMMARY
============================================================
Total questions evaluated: 12
  In-scope:  8
  Out-of-scope: 4

Retrieval Quality (in-scope only)
  Hit@3:    0.250
  MRR:       0.300
  Keyword Coverage: 0.200

Abstention Quality
  Abstention Accuracy:   0.000
  False Abstention Rate: 0.000
============================================================
```

### Reproducibility

- All random seeds are fixed (where applicable).
- The evaluation script loads the same `evaluation.jsonl` every time.
- Reports are saved as JSON (`baseline_evaluation_report.json`) for before/after comparison.
- **Important**: The `expected_doc_ids` in `evaluation.jsonl` are placeholders (DOC-01, DOC-02, etc.). After inspecting your actual `corpus.jsonl`, update them to match the real document IDs.

### Trade-offs

- **Hit@3 over Hit@1**: With sentence-aware chunking, a single document may produce 2–3 chunks. Hit@1 would unfairly penalize correct retrieval if the "best" chunk is not the one containing the exact answer. Hit@3 gives the re-ranker room to work.
- **No nDCG@K**: With only 16 documents and binary relevance, nDCG provides little additional signal over MRR + Hit@K. It was omitted to keep the framework simple and interpretable.
- **Keyword coverage as proxy for answer correctness**: Since no LLM generation is required, we cannot do exact-match answer grading. Keyword coverage checks that the retrieved text contains the expected facts.
