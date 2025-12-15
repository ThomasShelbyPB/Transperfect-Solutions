import argparse
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from challenge2.qe_utils import read_ch2_xlsx, compute_hter, corr_metrics, normalize_error_label, dump_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--embedder", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--clf_C", type=float, default=2.0)
    args = ap.parse_args()

    df = read_ch2_xlsx(args.xlsx)
    y = compute_hter(df["mt"], df["pe"])
    err = df["comment"].apply(normalize_error_label).tolist()
    texts = (df["src"] + "\n" + df["mt"]).tolist()

    enc = SentenceTransformer(args.embedder)
    src_emb = enc.encode(df["src"].tolist(), batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    mt_emb  = enc.encode(df["mt"].tolist(),  batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    cos = np.sum(src_emb * mt_emb, axis=1, keepdims=True)
    X_emb = np.hstack([cos, src_emb - mt_emb, src_emb, mt_emb]).astype(np.float32)

    clf = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95)),
        ("lr", LogisticRegression(max_iter=2000, C=args.clf_C, n_jobs=1)),
    ])
    clf.fit(texts, err)
    proba = clf.predict_proba(texts)

    X = np.hstack([X_emb, proba]).astype(np.float32)

    reg = Ridge(alpha=args.alpha)
    reg.fit(X, y)
    pred = reg.predict(X)

    metrics = corr_metrics(y, pred)

    df_out = df.copy()
    df_out["hter_proxy_true"] = y
    df_out["hter_proxy_pred"] = pred

    df_out.to_csv(args.out_csv, index=False)

    out = {
        "task": "challenge2_qe",
        "target": "hter_proxy",
        "model": "stacked(tfidf_logreg + ridge)",
        "embedder": args.embedder,
        "ridge_alpha": args.alpha,
        "clf_C": args.clf_C,
        "metrics_in_sample": metrics,
        "n_rows": int(len(df)),
        "predictions_csv": args.out_csv,
    }
    dump_json(args.out_json, out)
    print(out)


if __name__ == "__main__":
    main()
