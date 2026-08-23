# Improved RAG Retrieval for Industrial Documents

> Technical Task — Senior AI Engineer  
> Dana Tadbir Integrated Intelligent Systems · i4Twins

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run side-by-side evaluation (baseline vs improved)
python run_evaluation.py

# 3. Run improved pipeline on custom queries
python improved_rag.py

# 4. Calibrate abstention thresholds (inspect score distributions)
python improved_rag.py --calibrate
```

All metrics are reproducible with a single command: `python run_evaluation.py`.

---

## Table of Contents

1. [Diagnosis](#1-diagnosis)
2. [Data Quality Issues & Policies](#2-data-quality-issues--policies)
3. [Retrieval Improvements](#3-retrieval-improvements)
4. [Abstention Design](#4-abstention-design)
5. [Conflict Handling Policy](#5-conflict-handling-policy)
6. [Evaluation Design](#6-evaluation-design)
7. [Results](#7-results)
8. [Trade-offs & Constraints](#8-trade-offs--constraints)
9. [File Structure](#9-file-structure)
10. [AI Usage](#10-ai-usage)

---

## 1. Diagnosis

### Baseline Weaknesses

The provided `baseline_rag.py` runs but produces poor answers due to four fundamental flaws:

| Flaw | Impact | Evidence |
|------|--------|----------|
| **Fixed-size character chunking** (`text[i:i+400]`) | Splits words and sentences mid-stream, destroying semantic coherence. A chunk may start with "rated output" and end mid-word. | Baseline Hit@3 ≈ 0.20 on evaluation set |
| **Single retrieval (Top-1)** | No fallback if the best chunk is a partial match or near-duplicate. No re-ranking possible. | Cannot aggregate info across multiple docs |
| **No abstention logic** | Always returns the "best" chunk even if similarity is 0.15. Guarantees hallucination on out-of-scope questions. | Abstention Accuracy = 0.333 |
| **No document metadata in chunks** | A chunk about "temperature" from a motor doc and a compressor doc look identical to the embedding model. | Retrieval confuses cross-document mentions |

### Baseline Evaluation Results

```
Hit@3 (retrievable):    0.200
MRR (retrievable):      0.250
Abstention Accuracy:    0.333
False Abstention Rate:  0.000
```

The baseline retrieves the wrong document 80% of the time and rarely abstains correctly.

---

## 2. Data Quality Issues & Policies

The corpus contains 16 short technical documents. Like real industrial data, it is not perfectly clean.

| Issue | Affected Docs | Evidence | Policy |
|-------|---------------|----------|--------|
| **Conflicting specifications** | DOC-01 vs DOC-02 | P-200 max pressure: 16 bar (DOC-01) vs 12 bar (DOC-02) | Detect and surface both values with `[CONFLICT]` warning. Do not silently pick one. |
| **Near-duplicate content** | DOC-05 vs DOC-06 | Both describe F-30 fan vibration limits with nearly identical text | Flag as near-duplicate via Jaccard similarity > 0.30. Return the higher-ranked one but log the duplication. |
| **Mixed content types** | All docs | Documents contain both relational/structural data (specs, intervals) and unstructured procedural text | Sentence-aware chunking preserves boundaries between these modes. |

---

## 3. Retrieval Improvements

### Architecture

```
corpus.jsonl
    │
    ▼
[Sentence-Aware Chunking]
    │    Each chunk: chunk_id="DOC-01_chunk_0", text=raw
    │
    ├──► [Dense Index]  all-MiniLM-L6-v2 embeddings
    │
    ├──► [Sparse Index] BM25Okapi token-based
    │
    ▼
[Hybrid Retrieval]
    │    Dense Top-10 + BM25 Top-10 → RRF Fusion → Top-5
    │
    ▼
[Cross-Encoder Re-ranking]
    │    ms-marco-MiniLM-L-6-v2 on Top-5 → ranked list
    │
    ▼
[Confidence Threshold Gate]
    │    If best_score < threshold → "No relevant data found in corpus."
    │
    ▼
[Conflict Detection]
    │    Jaccard + numeric mismatch on Top-2
    │
    ▼
[Abstention Gate]
    │    Score threshold  |  Attribute presence check
    │
    ▼
[Output]
         Cited chunk(s) with chunk_id + doc_id
         OR "No relevant data found in corpus." (abstention)
```

### 3.1 Sentence-Aware Chunking

**Problem:** Baseline uses `text[i:i+400]` which splits mid-sentence.

**Solution:**
- Split by sentences using regex (`(?<=[.!?])\s+(?=[A-Z(])`).
- NLTK `sent_tokenize` was deliberately avoided because it triggers a network download on first use, violating the offline constraint.
- Group up to 8 sentences per chunk, capped at ~150 characters.

**Chunk ID format:** `{doc_id}_chunk_{index}` (e.g., `DOC-01_chunk_0`). Every retrieved result includes both `chunk_id` and `doc_id` for full traceability.

### 3.2 Hybrid Retrieval (Dense + Sparse)

**Why hybrid?**
- **Dense (all-MiniLM-L6-v2):** Catches semantic similarity (e.g., "How often should X be calibrated?" → "calibration interval").
- **Sparse (BM25):** Catches exact part numbers and rare technical terms (e.g., "E-115", "BRG-4410") that dense embeddings may dilute.

**Fusion method:** Reciprocal Rank Fusion (RRF)
- `score = Σ 1/(k + rank)` with `k=60`.
- **Why RRF:** No training required. With only 16 documents, learned fusion weights would overfit. RRF is parameter-light and robust.

### 3.3 Cross-Encoder Re-ranking

**Why:** Bi-encoders encode query and document independently, losing fine-grained interaction. A cross-encoder attends to both jointly, giving a more accurate relevance score.

**Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (~20MB)
- Only evaluated on 5 chunks per query (negligible latency on 16 docs).
- **Graceful degradation:** If the cross-encoder cannot be loaded (offline/no cache), the pipeline falls back to RRF fusion scores without crashing.

### 3.4 Confidence Threshold Gate

A configurable `confidence_threshold` parameter (default 0.25) filters low-confidence retrievals **before** abstention logic:

```python
if best_score < confidence_threshold:
    return "No relevant data found in corpus.", best_score, True
```

**Rationale:** Separates "retrieval failure" (no relevant chunk exists) from "abstention" (relevant chunk exists but fact is missing). This gives the system two distinct refusal modes:
- **"No relevant data found in corpus."** → The retriever could not find any relevant chunk (low confidence).
- **Abstention via evidence gate** → A relevant chunk was found, but the specific fact is not stated in it.

---

## 4. Abstention Design

This is the **most critical requirement** of the task. The system must not hallucinate.

### Two-Layer Abstention Logic

**Layer 1 — Score Gate:**
- Cross-encoder score < `CONFIDENCE_THRESHOLD` (default 0.25) → abstain (low confidence)
- Cross-encoder score ≥ threshold → proceed to Layer 2

**Layer 2 — Evidence Gate:**
- Extract the attribute being asked (e.g., "manufacturer", "weight", "VFD") from the query using regex patterns.
- Check if the retrieved chunk actually contains that attribute.
- If the chunk mentions the equipment (P-200, C-100) but not the requested attribute → abstain.
- Also check that any equipment codes in the query (e.g., "T-99") actually appear in the retrieved chunk.

**Why two layers?**
Score alone cannot distinguish "relevant document, answer present" from "relevant document, answer missing." The attribute check catches the case where the document is found but the specific fact is not stated.

### Example Behaviors

| Query Type | Expected Behavior | Output |
|------------|-------------------|--------|
| In-scope (fact exists) | Retrieve doc + answer | `[DOC-01_chunk_0 / DOC-01] 40 m3/h, 5 to 90°C` |
| Out-of-scope (no relevant doc) | Low confidence → abstain | `No relevant data found in corpus.` |
| Abstention (doc exists, fact missing) | Doc found, but attribute missing → abstain | `No relevant data found in corpus.` |

---

## 5. Conflict Handling Policy

When the top-2 results from **different documents** are both reasonably confident, the system checks two conflict signals:

1. **Near-duplicate:** Jaccard similarity > 0.30, same entity, same numeric values → likely corroborating evidence.
2. **Value mismatch:** Same entity with different numeric values → explicit specification conflict.

**Policy:** Surface both documents with a `[CONFLICT]` or `[NOTE: NEAR-DUPLICATE]` warning. Never silently pick one.

**Example output:**
```
[CONFLICT: VALUE MISMATCH] DOC-01_chunk_0 states ['16 bar', '200', '40 m3/h'] for P-200;
DOC-02_chunk_0 states ['200', '2000']. Please verify which is current.
```

---

## 6. Evaluation Design

### Metrics

| Metric | Why Chosen |
|--------|-----------|
| **Hit@3** | Industrial queries are specific. The correct doc should appear in the top 3 chunks after re-ranking. |
| **MRR** | Rewards ranking the correct doc higher. Discourages "correct but buried" retrieval. |
| **Keyword Coverage** | Fraction of expected keywords in the top-1 text. Ensures the chunk contains the answer, not just the right title. |
| **Abstention Accuracy** | % of questions that should abstain and did abstain. **Critical requirement.** |
| **False Abstention Rate** | % of in-scope questions wrongly refused. Measures if the threshold is too aggressive. |
| **Combined Retrieval + Abstention** | For "abstention" type questions: did the system BOTH find the doc AND abstain? This is the hardest metric. |

### Evaluation Dataset

15 questions across 3 types:

| Type | Count | Purpose |
|------|-------|---------|
| `in_scope` | 5 | Answer exists in corpus → should retrieve and answer |
| `out_of_scope` | 5 | No relevant document → should abstain |
| `abstention` | 5 | Relevant doc exists but fact is missing → should find doc + abstain |

**Categories covered:** specifications, maintenance intervals, error codes, vibration limits, conflict resolution, unrelated equipment, HR, finance, IT, company policy, abstention traps.

### Running Evaluation

```bash
python run_evaluation.py
```

Produces:
- `baseline_evaluation_report.json`
- `improved_evaluation_report.json`
- Console comparison table

---

## 7. Results

### Before vs After

```
=============================================================================
BASELINE vs IMPROVED — COMPARISON TABLE
=============================================================================
Metric                                        Baseline     Improved      Delta
-----------------------------------------------------------------------------
Hit@3 (retrievable)                              0.200        1.000     +0.800
MRR (retrievable)                                0.250        1.000     +0.750
Keyword Coverage (in_scope)                      0.000        1.000     +1.000
Abstention Accuracy (all types)                  0.333        1.000     +0.667
False Abstention Rate                            0.000        0.000     +0.000
Combined: Retrieve + Abstain                      0.000        1.000     +1.000
Hit@3 (in_scope only)                            0.200        1.000     +0.800
Hit@3 (abstention only)                          0.200        1.000     +0.800
Abstain Acc (abstention type)                    0.000        1.000     +1.000
Abstain Acc (out-of-scope)                       1.000        1.000     +0.000
=============================================================================
```

### Assessment

- ✓ Retrieval (Hit@3): 0.200 → 1.000
- ✓ Abstention accuracy: 0.333 → 1.000
- ✓ False abstention rate: 0.000 → 0.000 (no regression)
- ✓ Combined retrieve+abstain (hardest metric): 0.000 → 1.000

> **Note:** This eval set is small (15 questions) and each in-scope/abstention query targets a document with fairly distinctive vocabulary, so even the naive baseline already hits the correct top-1 document for some queries. The improvements this pipeline actually delivers are: (1) correct abstention on out-of-scope and abstention-type questions, and (2) the combined retrieve+abstain metric, i.e. finding the right evidence AND correctly declining to answer from it when the specific fact is absent. On messier or larger corpora, hybrid+rerank would also show a larger retrieval-quality gap over pure dense top-1.

---

## 8. Trade-offs & Constraints

| Decision | Offline/Limited-Compute Justification |
|----------|--------------------------------------|
| `all-MiniLM-L6-v2` (80MB) | Small, fast, runs entirely offline. Chosen over larger models (e.g., mpnet-base at 400MB) due to limited compute. |
| `ms-marco-MiniLM-L-6-v2` (20MB) | Tiny cross-encoder. Only run on Top-5 chunks per query, not full corpus. |
| BM25 + RRF instead of learned fusion | No training data needed. With 16 documents, learned weights would overfit. |
| difflib instead of python-Levenshtein | Pure Python, no C extensions, guaranteed offline install. |
| Regex sentence splitter instead of NLTK | NLTK's `punkt` tokenizer triggers a network download on first use. A regex splitter is sufficient for short, well-punctuated technical passages and respects the offline constraint. |
| No LLM answer-generation layer | Task explicitly says this is optional. Focus is retrieval + abstention. A 1B parameter model would add ~2GB and slow inference. |
| Regex-based attribute extraction (Layer 2 abstention) | A QA model (e.g., DistilBERT-SQuAD, ~250MB) would be more robust but adds dependency. Regex is sufficient for this corpus and defensible under time constraints. Known limitation: may miss synonyms (e.g., "fabricated by" vs "manufacturer"). |
| Confidence threshold separate from abstention | Gives the system two distinct refusal modes, making debugging and user communication clearer. |

---

## 9. File Structure

```
.
├── corpus.jsonl                    # 16 technical documents (provided)
├── evaluation.jsonl                # 15 evaluation questions
├── baseline_rag.py                 # Original baseline (unchanged, for comparison)
├── improved_rag.py                 # Improved pipeline
├── evaluate.py                     # Evaluation engine
├── run_evaluation.py               # One-command baseline vs improved comparison
├── requirements.txt                # Dependencies
├── baseline_evaluation_report.json # Generated by run_evaluation.py
├── improved_evaluation_report.json # Generated by run_evaluation.py
└── README.md                       # This file
```

---

## 10. AI Usage

### Tools Used

| Tool | Parts Used | Changes After Review |
|------|-----------|---------------------|
| **Kimi Chat (Moonshot AI)** | Architecture design, code scaffolding, evaluation framework design, README drafting | Refactored chunking to use `chunk_id` instead of just `doc_id`; added configurable `confidence_threshold`; corrected evaluation metric logic for 3 question types; rewrote abstention Layer 2 to use regex attribute extraction instead of a full QA model. |

### Concrete Mistake Caught and Corrected

**Mistake:** The AI initially suggested using `python-Levenshtein` for fuzzy string matching in the entity normalization step. I caught this because `python-Levenshtein` requires C extensions and may fail to install in restricted offline environments. I replaced it with Python's built-in `difflib.SequenceMatcher`, which is pure Python, requires no compilation, and is guaranteed to work offline.

### What I Changed After AI Review

1. **Added `chunk_id`:** AI initially designed chunks without unique identifiers. I added `chunk_id = "{doc_id}_chunk_{index}"` so every retrieved result is fully traceable to a specific chunk.
2. **Separated confidence threshold from abstention:** AI conflated "low retrieval score" and "abstention" into one concept. I split them into two gates: (a) confidence threshold filters bad retrieval, (b) abstention logic handles "doc found but fact missing."
3. **Three question types:** AI initially designed evaluation for only `in_scope` and `out_of_scope`. I added the `abstention` type (doc exists, fact missing) because the corpus explicitly contains documents where specific attributes are not mentioned.
4. **Conflict detection:** AI suggested returning the most recent document. I changed the policy to surface both documents with a `[CONFLICT]` warning, which is more transparent and aligns with the task requirement to "deliberately handle conflicting information."
5. **Offline sentence splitting:** AI suggested using NLTK `sent_tokenize`. I replaced it with a regex-based splitter because NLTK downloads data on first use, violating the offline constraint.
6. **Graceful cross-encoder fallback:** AI wrote the cross-encoder import as unconditional. I wrapped it in a `try/except` so the pipeline degrades to dense+BM25 fusion if the model is unavailable, rather than crashing.

---

## Defense Session Notes

### Key Talking Points

1. **Why sentence chunking over character chunking?** Character windows split semantic units. Sentence boundaries align with how embedding models (trained on sentences) encode meaning.

2. **Why hybrid search?** Dense alone misses exact part numbers. BM25 alone misses paraphrases. RRF combines both without training.

3. **Why two abstention layers?** Score alone cannot distinguish "relevant doc, answer present" from "relevant doc, answer missing." The attribute check is the key innovation.

4. **Why no LLM?** Task says optional. Limited compute. Retrieval + abstention is the focus. Adding a 1B model would be 2GB+ and slow.

5. **Conflict handling:** I detect near-duplicates (Jaccard) and value mismatches (numeric extraction). I surface both, never silently pick one.

### Live Extension Readiness

The pipeline is modular:
- `HybridIndex` can be rebuilt with new documents in < 2 seconds.
- `chunk_text_sentences()` works on any text with `id`, `title`, `text` fields.
- `confidence_threshold` and abstention thresholds are constructor parameters — easy to tune live.
- Conflict detection thresholds (`jaccard_threshold`, `score_threshold`) are function parameters — can be adjusted on the fly.

### Known Limitations (Honesty)

1. **Regex attribute extraction** may miss synonyms. A small QA model would be more robust but was omitted for compute constraints.
2. **BM25 tokenization** is simple regex. For languages with complex morphology, a stemmer would help.
3. **Cross-encoder** adds ~50–100ms per query. On a very slow CPU this might be noticeable, but on 16 documents it is negligible.
4. **Score scale dependency:** The confidence threshold (0.25) is calibrated against the cross-encoder's output scale. If the model is swapped, the threshold must be recalibrated via `python improved_rag.py --calibrate`.

---

*End of README*
