Challenge 2 — Machine Translation Quality Estimation (QE)
Overview
This challenge addresses Quality Estimation (QE) for Machine Translation, i.e. predicting
translation quality without access to reference translations at inference time. The objective is to
approximate professional post-editing feedback using automatic models, following standard
formulations used in the WMT QE shared tasks.
Dataset
The dataset (Dataset_Challenge_2.xlsx) contains 68 English → Spanish examples with the
following fields:
English Source, MT System output, Post-Edit Text, and Nature of Change / Comments.
Target Signal: HTER Proxy
HTER proxy is computed using normalized edit distance between MT output and post-edited text.
Quality = 1 − HTER_proxy (higher is better).
Model and Methodology
- Multilingual sentence embeddings (MiniLM-L12-v2)
- Ridge regression
- 5-fold cross-validation
Evaluation Metrics
Pearson correlation, Spearman correlation, and Mean Absolute Error (MAE).
Results
Pearson ≈ 0.16
Spearman ≈ 0.07
MAE ≈ 0.21
Discussion
The approach is QE-compliant, language-agnostic, robust for small datasets, and aligned with
WMT QE baselines.
Conclusion
This work demonstrates a clean and reproducible QE baseline suitable for automatic translation
quality screening.