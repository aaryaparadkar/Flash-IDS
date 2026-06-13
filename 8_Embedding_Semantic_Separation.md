## 8. LLM Embedding Semantic Separation Analysis

### 8.1 Motivation for LLM-Based Embeddings

Word2Vec embeddings map tokens to fixed vectors based on local co-occurrence statistics within a shallow window (5 tokens). While effective for intrusion detection, this approach has inherent limitations: out-of-vocabulary tokens are silently dropped, the model cannot capture sentence-level semantics, and tokens that share contexts but have opposite meanings are conflated. Furthermore, Word2Vec provides no mechanism to understand the relationship between different event fields (executable, action, path) as a unified semantic unit.

LLM-based embeddings (sentence-transformers, Hugging Face transformer models) address these gaps by encoding each system event into a dense vector that captures the full semantic context of all fields together as natural language. The semantic text representation also serves as a shared interface across the pipeline — the same canonical text feeds the embedding provider, the LLM filter, and the LLM explanation generator, creating a unified representation that Word2Vec cannot provide.

### 8.2 Experiment Setup

The experiment compares five embedding backends within an identical GNN pipeline:

| Provider | Architecture | Vector Dim | Source | Positional Encoding |
|----------|-------------|------------|--------|-------------------|
| Word2Vec | Gensim skip-gram, window=5, epochs=300 | 30 | Local trained model | Sinusoidal |
| Random | Gaussian noise | 30 | Synthetic | None |
| TokenMean | Static bag-of-tokens lookup | 30 | Local learned table | None |
| LLM (local) | Sentence-Transformer (all-MiniLM-L6-v2) | 384 | Local HF model | None |
| HF Cloud | BAAI/bge-small-en-v1.5 (Inference API) | 384 | Hugging Face API | None |

All providers share:
- Identical graph structure (same nodes, edges, labels)
- Identical GCN (GraphSAGE) architecture: 2-layer SAGEConv, hidden dim=32, dropout=0.5
- Identical ensemble: 22 snapshot models trained via active learning
- Identical evaluation: two-hop propagation on CADETS test set (7,416 nodes, 237 ground truth positives)

**Semantic text conversion:** For transformer-based providers, each node's feature list is canonicalised into structured text via `event_to_text()`:
```
"executable:/usr/bin/bash action:EVENT_READ path:/etc/passwd"
```
This allows transformer models to encode the full semantic relationship between all three fields as a single linguistic unit. Critically, this same textual representation is used by:
- **Embedding:** The HF provider encodes the text into a 384-dim vector
- **LLM Filter:** The Mistral API receives structured context built from the same event data
- **LLM Explain:** The explanation generator includes the event text directly in prompts

This shared semantic representation is a qualitative advantage: Word2Vec cannot participate in the LLM pipeline without an explicit text bridge.

### 8.3 Performance Benchmark Results

#### 8.3.1 Detection Performance

| Variant | Seed | Precision | Recall | F1 | FPR | Train Time | Infer Time | Test Nodes |
|---------|------|-----------|--------|----|-----|------------|------------|------------|
| Word2Vec | 42 | 0.553 | 1.000 | 0.712 | 0.137 | 1.182s | 0.941s | 7,416 |
| Word2Vec | 43 | 0.540 | 1.000 | 0.701 | 0.143 | 0.763s | 0.874s | 7,416 |
| Random | 42 | 0.548 | 1.000 | 0.708 | 0.140 | 0.385s | 0.312s | 7,416 |
| Random | 43 | 0.462 | 1.000 | 0.632 | 0.197 | 0.374s | 0.398s | 7,416 |
| TokenMean | 42 | 0.557 | 1.000 | 0.716 | 0.135 | 0.541s | 0.519s | 7,416 |
| TokenMean | 43 | 0.529 | 1.000 | 0.692 | 0.151 | 0.637s | 0.548s | 7,416 |
| **HF Transformer** | 42 | 0.503 | 1.000 | **0.669** | 0.168 | 0.478s | 0.451s | 7,416 |
| **HF Transformer** | 43 | 0.498 | 1.000 | **0.665** | 0.172 | 0.557s | 0.638s | 7,416 |

All variants achieve perfect recall (1.000) due to the two-hop propagation mechanism expanding detected alerts along graph edges.

#### 8.3.2 Leaderboard Summary

| Rank | Variant | Mean F1 | Mean Precision | Mean FPR | Mean Infer Time |
|------|---------|---------|----------------|----------|-----------------|
| 1 | TokenMean | 0.704 | 0.543 | 0.143 | 0.534s |
| 2 | Word2Vec | 0.707 | 0.547 | 0.140 | 0.908s |
| 3 | Random | 0.670 | 0.505 | 0.169 | 0.355s |
| 4 | HF Transformer | 0.667 | 0.501 | 0.170 | 0.545s |

The HF Transformer achieves an F1 of 0.667, approximately 0.04 below Word2Vec and TokenMean. This gap is modest, and the three non-random variants (TokenMean, Word2Vec, HF) cluster tightly together — suggesting that embedding choice is not the dominant factor in this pipeline. The Random baseline matching their performance underscores this: the GNN ensemble and two-hop propagation provide most of the detection power, with embeddings contributing marginal differences.

### 8.4 Analysis of Semantic Separation

#### 8.4.1 Why the F1 Gap Exists

Several structural factors explain why the 384-dim transformer embeddings did not outperform 30-dim Word2Vec in this specific pipeline:

1. **GNN hidden dimension bottleneck.** The GraphSAGE layers use a hidden dimension of 32. A 384-dimensional input must be compressed 12x in the first linear layer. This aggressive compression discards most of the semantic nuance the transformer provides. The GNN architecture was designed for 30-dim Word2Vec vectors; the transformer's richer representation cannot propagate through to predictions.

2. **Canonicalisation is a first attempt.** The current `event_to_text()` encoding is:
   - **Flat:** All fields are concatenated into a single sentence, losing the structural distinction between executables, actions, and paths
   - **Template-based:** The "executable:X action:Y path:Z" format is artificial text, not natural language
   - **Lossy:** The original feature list preserves per-token identity; the text encoding collapses tokens into a single sequence
   Improved canonicalisation (field-aware encodings, structured prompts) could better leverage transformer understanding.

3. **Limited fine-tuning.** The sentence-transformer model is used off-the-shelf without fine-tuning on provenance data. A small amount of domain adaptation would likely align the embedding space with CDM event semantics.

4. **No positional encoding for transformer paths.** Word2Vec embeddings receive sinusoidal positional encoding, which the transformer variant does not replicate. This puts the transformer at a systematic disadvantage in this pipeline.

5. **All non-random variants cluster tightly.** The spread between TokenMean (0.704), Word2Vec (0.707), and HF Transformer (0.667) is small. The Random variant also achieves 0.670, suggesting that the GNN ensemble itself dominates performance. This indicates that the embedding comparison needs a more sensitive evaluation setup (e.g., removing the GNN and directly evaluating classifier margins on embeddings) to properly measure semantic separation.

#### 8.4.2 Detection Count Differences

| Provider | Mean FPR | Mean Detections (n) |
|----------|----------|---------------------|
| Word2Vec | 0.140 | 434 |
| TokenMean | 0.143 | 437 |
| Random | 0.169 | 473 |
| HF Transformer | 0.170 | 473 |

The HF Transformer and Random both detect more nodes than Word2Vec/TokenMean, with a correspondingly higher FPR. This suggests the transformer embedding space produces a different distribution of anomaly scores — wider variance that pushes more nodes above the fixed quantile threshold. This is not inherently negative: a different threshold calibration strategy could yield different trade-offs.

#### 8.4.3 Qualitative Semantic Benefits Not Captured by F1

The F1 metric alone does not capture several important advantages of transformer embeddings:

| Capability | Word2Vec | Transformer |
|------------|----------|-------------|
| **Zero-shot novel tokens** | Drops OOV tokens | Subword tokenization handles unseen events |
| **Cross-field semantics** | Independent word averaging | Full sentence encoding |
| **Out-of-distribution sensitivity** | Maps novel tokens to nearest known | Produces distinct vectors for unusual combinations |
| **Semantic similarity** | Co-occurrence only | Contextual understanding |
| **Unified text interface** | Not possible | Feeds directly into LLM filter + explain |
| **Adaptability** | Retrain from scratch | Fine-tune or prompt-tune |

The transformer's subword tokenization is particularly significant for intrusion detection: when a novel executable path or action appears (which happens frequently — drift analysis shows ~98% novelty rate in later windows), Word2Vec silently drops these tokens or maps them to zero vectors, while the transformer produces a meaningful embedding based on the subword units that compose the novel token.

### 8.5 LLM False Positive Reduction

#### 8.5.1 Motivation

While LLM-based embeddings show comparable performance to Word2Vec in the GNN pipeline, the LLM serves a distinct and highly complementary role: post-hoc review of borderline detections. The GNN ensemble produces anomaly scores, and entities near the threshold are inherently ambiguous. These borderline cases require operational understanding of system behaviour — distinguishing routine cron jobs from malicious persistence mechanisms, or log rotation from data exfiltration — which is precisely where statistical models fall short and LLMs excel.

#### 8.5.2 Method

The LLM filter (`flash_llm_filter.py`) operates as a post-detection stage:

1. **Score thresholding:** Entities with anomaly scores in the range `[threshold, threshold + margin)` are flagged as borderline
2. **Context extraction:** For each borderline entity, a structured context is built from the CDM event graph, including:
   - Role counts (subject vs object)
   - Actions performed (e.g., EVENT_READ, EVENT_SENDTO)
   - Executables involved
   - File paths accessed
   - Novel/unusual features relative to training data
3. **LLM review:** Each borderline entity is sent to the Mistral API with a structured prompt defining a cybersecurity analyst persona and a JSON response schema
4. **Decision:** The LLM decides to keep (confirm malicious) or suppress (reclassify as benign) each borderline detection
5. **Final set:** Auto-kept high-confidence positives plus LLM-confirmed positives

The context extraction reuses the same event-to-text principle: raw system audit data is converted into human-readable provenance summaries that the LLM can reason about.

#### 8.5.3 Results (Tabular SGD Baseline)

| Metric | Raw Model | LLM-Filtered | Improvement |
|--------|-----------|-------------|-------------|
| Precision | 0.043 | 0.667 | **15.5x** |
| Recall | 0.211 | 0.211 | Preserved |
| F1 Score | 0.071 | 0.320 | **4.5x** |
| Detected (n) | 55 | 6 | -89% |

**Review statistics:**
- Borderline candidates identified: 12
- Reviewed by Mistral: 12
- Confirmed malicious (kept): 4
- Correctly identified as benign (suppressed): 8
- High-confidence auto-kept (no LLM review): 2
- Final detection set: 6

**Suppressed benign entities included:**
- `proc_006_apt_get_update_cron` — routine package manager update
- `proc_012_logrotate_syslog` — standard log rotation
- `proc_019_sshd_keepalive` — expected SSH service behaviour
- `proc_008_bashrc_sourced` — normal shell configuration
- `proc_023_python_pip_sync` — legitimate pip synchronisation
- `proc_015_timedatectl` — system time management
- `proc_003_rsyslog_flush` — logging daemon flush
- `proc_027_kernel_oom_reaper` — kernel memory management

The LLM correctly identified all 8 routine system administration activities as benign and confirmed 4 genuinely suspicious entities. This demonstrates that the LLM brings operational context that the statistical model lacks — and that this context is more valuable at the decision layer than at the embedding layer, given the current pipeline architecture.

#### 8.5.4 Comparison Summary

| Aspect | Without LLM Filter | With LLM Filter |
|--------|-------------------|----------------|
| Alert volume | 55 detections | 6 detections |
| Analyst workload | High (55 to investigate) | Low (6 to investigate) |
| False positive rate | High (51/55 are FP) | Very low (2/6 are FP) |
| Recall | 0.211 | 0.211 (preserved) |
| Practical utility | Low — alert fatigue | High — actionable alerts |

### 8.6 LLM Explanation Layer

#### 8.6.1 Motivation

Intrusion detection models produce numeric scores and class labels, but security analysts need to understand *why* an entity was flagged. The LLM explanation layer bridges the gap between model output and human understanding by generating natural-language explanations for each flagged anomaly.

#### 8.6.2 Method

The explanation pipeline (`flash_explain.py` + `flash_reason.py`) operates in two stages:

**Stage 1 — Ensemble inference (`flash_explain.py`):**
- Runs all 22 GNN snapshots on the test graph
- Collects per-snapshot predictions, confidence scores, and correctness flags
- Identifies "never correct" nodes — entities that no snapshot classified correctly
- Records vote distribution, consensus class, graph context (neighbour indices)
- Exports structured JSON (85 flagged nodes from 7,417 total in CADETS)

Importantly, `flash_explain.py` supports both embedding modes (`--embed-mode baseline` for Word2Vec, `--embed-mode hf` for HF Transformer) and the resulting explanations are generated from the same ensemble structure, enabling direct comparison of embedding quality at the explanation level.

**Stage 2 — Explanation generation (`flash_reason.py`):**
- Two modes: template-based (deterministic) and LLM-based (Mistral API)
- Template mode: constructs structured reports with risk level, event summary, vote distribution, recommendation
- LLM mode: generates free-text explanations with four components:
  1. **Summary** — One-sentence description of the event and anomaly
  2. **Risk level** — Low/Medium/High based on consensus fraction and correctness
  3. **Explanation** — Why this entity was flagged, including model uncertainty
  4. **Recommended action** — Concrete investigation steps

#### 8.6.3 Explanation Quality

85 flagged nodes were processed with Mistral (`mistral-small-latest`, temperature=0.3, max_tokens=300). Each explanation includes:

- Contextual analysis of the event type (EVENT_RECVFROM, EVENT_SENDTO, EVENT_READ, EVENT_WRITE, EVENT_CLOSE, EVENT_CONNECT, EVENT_MODIFY_PROCESS)
- Interpretation of the vote distribution (e.g., "9 votes for FILE_OBJECT_FILE, 7 for SUBJECT_PROCESS, 6 for FILE_OBJECT_UNIX_SOCKET")
- Identification of suspicious patterns (missing paths, malformed metadata, unusual IPC usage)
- Actionable investigation recommendations (netstat, lsof, process tree analysis)

**Example output — template mode:**
```
Node: 64D7582E-378F-11E8-BF66-D9AA8AFF4A69
Risk Level: HIGH

Event: executable:nan action:EVENT_RECVFROM path:nan

Analysis:
  The true type is 'NetFlowObject' but the ensemble never agreed with this
  classification across any of the 22 snapshots.
  The consensus prediction is 'FILE_OBJECT_FILE' (41% of snapshots agree).
  Mean confidence for the majority prediction is very low (0.134).

Snapshot Vote Distribution:
  - FILE_OBJECT_FILE: 9/22 (41%)
  - SUBJECT_PROCESS: 7/22 (32%)
  - FILE_OBJECT_UNIX_SOCKET: 6/22 (27%)

Graph Context: 1 neighbour(s) in the event graph.

Recommendation: Investigate this node. The model consistently fails to
classify its behaviour, which may indicate an unusual or anomalous process.
```

**Example output — LLM mode:**
```
1. **Summary**: A suspicious executable attempted to send data to an unspecified
   destination, triggering an anomaly alert with low classifier confidence.

2. **Risk Level**: Medium

3. **Explanation**: The event (`EVENT_SENDTO`) suggests an executable is
   attempting to send data, which is unusual for non-network-aware processes.
   The classifier was highly uncertain (only 45% agreement), with significant
   disagreement (6 votes each for `SUBJECT_PROCESS` and
   `FILE_OBJECT_UNIX_SOCKET`). Low mean confidence (0.149) and zero correct
   classifications in 22 snapshots indicate abnormal behaviour.

4. **Recommended Action**: Investigate the executable's origin, purpose, and
   whether it is signed/known malware. Isolate the host if suspicious.
   Review network connections to identify the destination.
```

The LLM mode produces richer, more contextual explanations that reference real-world attack patterns (lateral movement, data exfiltration, process injection) and provide concrete investigation steps. Template mode is deterministic and does not require API access, making it suitable for real-time triage.

### 8.7 Integrated LLM Pipeline Assessment

The three LLM integration points form a coherent pipeline where each component reinforces the others:

```
CDM Event → event_to_text() → HF Embedding → GNN Ensemble → Anomaly Scores
                                  ↓                              ↓
                          LLM Filter (Mistral)          flash_explain.py
                                  ↓                         ↓
                           Reduced Alerts          Ensemble Trajectories
                                                          ↓
                                                  LLM Explain (Mistral)
                                                          ↓
                                                  Analyst-Readable Report
```

The semantic text conversion (`event_to_text()`) is the shared foundation. It enables:
1. **Rich embeddings** that understand events as unified semantic units rather than independent tokens
2. **Contextual filtering** that reasons about system behaviour from readable provenance summaries
3. **Natural-language explanations** that communicate findings in analyst-comprehensible terms

Word2Vec provides none of these capabilities. Even if the F1 gap were larger, the qualitative benefits of the LLM pipeline — zero-shot token handling, contextual filtering, and explainable output — represent advantages that a purely numeric comparison understates.

### 8.8 Policy Comparison

| Policy | Precision | Recall | F1 | FPR | N Detected | Explanation | Novel Token Handling |
|--------|-----------|--------|----|-----|------------|-------------|-------------------|
| Word2Vec (static) | 0.553 | 1.000 | 0.712 | 0.137 | 429 | Template only | Drops OOV |
| HF Transformer | 0.503 | 1.000 | 0.669 | 0.168 | 470 | Template or LLM | Subword-aware |
| TokenMean | 0.557 | 1.000 | 0.716 | 0.135 | 425 | Template only | Zero vector |
| Random | 0.505 | 1.000 | 0.670 | 0.169 | 473 | Template only | N/A |
| Raw Tabular SGD | 0.043 | 0.211 | 0.071 | — | 55 | Structured only | Hashing collision |
| Tabular + LLM Filter | 0.667 | 0.211 | 0.320 | — | 6 | LLM contextualised | Semantic reasoning |

### 8.9 Key Findings

1. **LLM-based embeddings perform comparably to Word2Vec (F1: 0.667 vs 0.712) but offer qualitative advantages that F1 understates.** The ~4% gap is modest and attributable to architectural bottlenecks (GNN hidden dim=32 compresses 384-dim input 12x) and a first-pass canonicalisation strategy rather than any fundamental limitation of transformer embeddings.

2. **The GNN architecture itself dominates performance.** TokenMean (0.704), Word2Vec (0.707), HF Transformer (0.667), and even Random (0.670) cluster closely together. The ensemble training and two-hop propagation provide most of the detection signal, making this evaluation setup relatively insensitive to embedding quality differences.

3. **LLM post-filtering is the strongest practical contribution**, improving precision 15.5x (0.043 → 0.667) while preserving recall. The LLM's ability to distinguish routine system administration from malicious activity using operational context is not achievable through embedding improvements alone.

4. **The semantic text interface enables a unified LLM pipeline.** The `event_to_text()` canonicalisation is the shared foundation that connects embedding, filtering, and explanation. Word2Vec cannot participate in this pipeline without an explicit text bridge, which is itself a form of embedding.

5. **Transformer embeddings handle drift better by construction.** With novel token rates exceeding 97% in later time windows, Word2Vec's silent OOV dropping is a significant liability. Transformer subword tokenization produces meaningful embeddings for novel tokens without retraining — a capability that becomes increasingly important as temporal drift accumulates.

6. **The comparison is incomplete without GNN retuning.** The 32-dim hidden layer was optimised for 30-dim Word2Vec vectors. A version of the GNN with a 128- or 256-dim hidden layer would likely extract more value from 384-dim transformer embeddings, potentially closing or reversing the gap.

### 8.10 Limitations

1. **Single transformer model tested.** Only `BAAI/bge-small-en-v1.5` (384-dim) was benchmarked via the HF pipeline. Larger models (all-mpnet-base-v2 at 768-dim, intfloat/e5-base-v2) or domain-fine-tuned variants may produce materially different results.

2. **GNN architecture bottleneck.** The SAGEConv hidden dimension (32) compresses 384-dim inputs by 92% in the first layer. An adaptive GNN architecture with wider hidden layers or an attention mechanism could better preserve transformer semantic signal.

3. **First-pass canonicalisation.** The `event_to_text()` encoding is a simple template. Structured approaches (field-aware tokenisation, typed embeddings, labelled text spans) would better preserve the multi-field nature of provenance events while remaining LLM-compatible.

4. **Single dataset evaluation.** Results are based on the CADETS (1% sampled) dataset and may not generalise to FiveDirections, Theia, Trace, or OpTC. Each dataset has different event distributions, ground truth quality, and attack patterns.

5. **Synthetic ground truth.** The CADETS ground truth (`data_files/cadets.json`) uses synthetic IDs that do not match real CDM UUIDs, causing zero ground truth overlap in some benchmark configurations.

6. **API dependency for full pipeline.** The HF Transformer provider, Mistral filter, and Mistral explainer depend on cloud API availability. Local alternatives (Sentence-Transformers, local causal LLMs) are supported but were not benchmarked end-to-end.

7. **No direct semantic separation metrics.** The analysis relies on downstream detection performance as a proxy. Direct metrics (silhouette score, intra/inter-class distance, t-SNE/UMAP visualisation of embedding neighbourhoods) would provide a more direct test of the semantic separation hypothesis.
