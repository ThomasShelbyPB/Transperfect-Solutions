import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from challenge2.qe_utils import (
    read_ch2_xlsx, compute_hter, corr_metrics, normalize_error_label, dump_json
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--embedder", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--clf_C", type=float, default=2.0)
    args = ap.parse_args()

    df = read_ch2_xlsx(args.xlsx)

    # labels
    y = compute_hter(df["mt"], df["pe"])

    # coarse error labels from comments
    err = df["comment"].apply(normalize_error_label).tolist()

    # embedder features
    enc = SentenceTransformer(args.embedder)
    src_emb = enc.encode(df["src"].tolist(), batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    mt_emb  = enc.encode(df["mt"].tolist(),  batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    cos = np.sum(src_emb * mt_emb, axis=1, keepdims=True)
    X_emb = np.hstack([cos, src_emb - mt_emb, src_emb, mt_emb]).astype(np.float32)

    # classifier features (fast)
    texts = (df["src"] + "\n" + df["mt"]).tolist()

    clf = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95)),
        ("lr", LogisticRegression(max_iter=2000, C=args.clf_C, n_jobs=1)),
    ])

    # K-fold: train clf on train fold, generate probs for train/test, train ridge
    kf = KFold(n_splits=args.k, shuffle=True, random_state=42)
    folds = []

    labels_sorted = sorted(set(err))

    for tr, te in kf.split(X_emb):
        clf.fit([texts[i] for i in tr], [err[i] for i in tr])

        # predict_proba returns columns in clf.classes_ order
        proba_tr = clf.predict_proba([texts[i] for i in tr])
        proba_te = clf.predict_proba([texts[i] for i in te])

        # stack: [embedding feats | error probs]
        X_tr = np.hstack([X_emb[tr], proba_tr]).astype(np.float32)
        X_te = np.hstack([X_emb[te], proba_te]).astype(np.float32)

        reg = Ridge(alpha=args.alpha)
        reg.fit(X_tr, y[tr])
        pred = reg.predict(X_te)

        folds.append(corr_metrics(y[te], pred))

    def agg(key):
        vals = np.array([m[key] for m in folds], dtype=np.float32)
        return {"mean": float(vals.mean()), "std": float(vals.std())}

    out = {
        "task": "challenge2_qe",
        "target": "hter_proxy",
        "model": "stacked(tfidf_logreg + ridge)",
        "embedder": args.embedder,
        "ridge_alpha": args.alpha,
        "clf_C": args.clf_C,
        "kfold": args.k,
        "error_labels_used": labels_sorted,
        "metrics_kfold": {
            "pearson": agg("pearson"),
            "spearman": agg("spearman"),
            "mae": agg("mae"),
        },
        "n_rows": int(len(df)),
    }

    dump_json(args.out_json, out)
    print(out)


if __name__ == "__main__":
    main()
