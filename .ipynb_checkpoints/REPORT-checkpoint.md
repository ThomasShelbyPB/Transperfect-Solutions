# TransPerfect Machine Translation & Quality Estimation

This repository presents an end-to-end solution for the **TransPerfect MT & QE Technical Challenge**, covering:

* **Challenge 1**: Machine Translation fine-tuning and evaluation
* **Challenge 2**: Quality Estimation (QE) without reference translations

The emphasis is on **robust modeling choices, interpretability, and reproducibility**, while excluding large artifacts (model checkpoints, logs) to keep the submission lightweight.

---

## Repository Structure

```
.
├── challenge1/                 # Machine Translation
│   ├── train_seq2seq_pl.py
│   ├── train_causal_lora.py
│   ├── eval_mt.py
│   └── common_metrics.py
│
├── challenge2/                 # Quality Estimation
│   ├── qe_train_ridge_cv.py
│   ├── qe_train_stacked_cv.py
│   ├── qe_predict.py
│   ├── qe_utils.py
│   └── readme.md
│
├── data/                       # Provided datasets (optional)
│   ├── Dataset_Challenge_1.xlsx
│   └── Dataset_Challenge_2.xlsx
│
├── outputs-challenge1/         # JSON results only
├── REPORT.md                   # Detailed QE explanation
├── requirements.txt
└── Untitled.ipynb              # Optional exploration
```

Large files such as `.ckpt`, `.safetensors`, `lightning_logs/`, and `.venv/` are intentionally excluded.

---

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Challenge 1 — Machine Translation

### Objective

Fine-tune a neural MT model and evaluate translation quality on both a **general benchmark** and a **domain-specific software UI dataset**.

### Model

* **Helsinki-NLP MarianMT (EN → NL)**
* Encoder–decoder Transformer
* Fine-tuned using PyTorch Lightning

### Training (Example)

```bash
python challenge1/train_seq2seq_pl.py \
  --model_name Helsinki-NLP/opus-mt-en-nl \
  --src_lang en --tgt_lang nl \
  --batch_size 8 \
  --max_length 128 \
  --max_steps 600 \
  --lr 1e-5 \
  --output_dir outputs/seq2seq_marian_fast
```

### Evaluation

```bash
python challenge1/eval_mt.py \
  --model_dir outputs/seq2seq_marian_fast \
  --model_type seq2seq \
  --flores_lang_pair eng_Latn-nld_Latn \
  --software_test_xlsx data/Dataset_Challenge_1.xlsx \
  --out_json outputs/eval_seq2seq_marian_fast.json
```

### Metrics

* **chrF (proxy)** — character-level adequacy (higher = better)
* **HTER (proxy)** — edit distance to post-edit (lower = better)

Results indicate stable domain adaptation with small but consistent gains on software UI text.

---

## Challenge 2 — Quality Estimation (QE)

### Objective

Predict MT quality **without reference translations**, approximating feedback from professional linguists.

---

### Dataset

Each segment contains:

* English source
* MT system output (Spanish)
* Human post-edit
* Linguist comments

Dataset size: **68 segments**

---

### Target Signal (HTER Proxy)

Because gold HTER labels are unavailable, a proxy is computed using normalized edit distance:

```
HTER_proxy = 1 − SequenceMatcher(mt, pe).ratio()
quality    = 1 − HTER_proxy
```

* `quality ∈ [0, 1]`
* Higher values indicate better MT output

---

### Features

* Sentence embeddings of MT output
* Model: `paraphrase-multilingual-MiniLM-L12-v2`
* Embedding dimension: 384

---

### Model

* **Ridge Regression**
* Selected for:

  * Stability on very small datasets
  * Explicit regularization control
  * Interpretability

---

### QE Pipeline

```
MT Output
   │
   ▼
Sentence Embedding
   │
   ▼
Ridge Regression
   │
   ▼
Predicted Quality Score
```

---

### Training & Evaluation

```bash
python challenge2/qe_train_ridge_cv.py \
  --xlsx data/Dataset_Challenge_2.xlsx \
  --out_json outputs/ch2_qe_results.json \
  --k 5 \
  --alpha 10
```

* 5-fold cross-validation
* Metrics reported as **mean ± standard deviation**

---

### Metrics Interpretation

| Metric   | Description                          |
| -------- | ------------------------------------ |
| Pearson  | Linear correlation with true quality |
| Spearman | Rank correlation (quality ordering)  |
| MAE      | Absolute prediction error            |

Observed behavior:

* **Low MAE** → reasonable calibration
* **Modest correlations** → expected due to:

  * Small dataset size
  * Diverse linguistic error types
  * High variance across folds

This behavior aligns with findings from prior WMT QE tasks under low-resource conditions.

---

## Conclusion

This submission demonstrates:

* Practical MT fine-tuning with domain adaptation
* A principled QE approach under limited supervision
* Clear trade-offs between simplicity, robustness, and interpretability

All scripts are runnable end-to-end, results are stored as lightweight JSON files, and the overall design reflects real-world localization and MT quality workflows.

---

## Notes for Reviewers

* Heavy artifacts are excluded by design
* All reported metrics are reproducible
* The approach prioritizes **engineering realism over leaderboard optimization**
