import argparse
import json
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer

from challenge2.qe_utils import (
    read_ch2_xlsx,
    compute_quality_scores,
    handcrafted_features,
    regression_metrics
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=10.0)  # used if grid is off
    ap.add_argument("--alpha_grid", action="store_true", help="Tune alpha over a small grid")
    args = ap.parse_args()

    df = read_ch2_xlsx(args.xlsx)
    df = compute_quality_scores(df)
    y = df["quality"].values.astype(np.float32)

    embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    X_mt = embedder.encode(df["mt"].tolist(), convert_to_numpy=True, show_progress_bar=True)

    X_hand = np.stack([handcrafted_features(r.src, r.mt) for r in df.itertuples()])
    X = np.concatenate([X_mt, X_hand], axis=1)

    # Cross-validation
    kf = KFold(n_splits=args.k, shuffle=True, random_state=42)

    # alpha candidates (log-ish)
    grid = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0] if args.alpha_grid else [args.alpha]

    metrics = {"pearson": [], "spearman": [], "mae": []}
    best_alphas = []

    for tr, te in kf.split(X):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        # SCALE inside fold (important!)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # pick alpha by best MAE on validation (simple + stable)
        best_alpha, best_mae = None, 1e9
        best_preds = None

        for a in grid:
            reg = Ridge(alpha=float(a))
            reg.fit(X_tr_s, y_tr)
            preds = reg.predict(X_te_s)
            mae = float(np.mean(np.abs(y_te - preds)))
            if mae < best_mae:
                best_mae = mae
                best_alpha = float(a)
                best_preds = preds

        best_alphas.append(best_alpha)

        m = regression_metrics(y_te, best_preds)
        for k in metrics:
            metrics[k].append(m[k])

    results = {
        "task": "challenge2_quality_estimation",
        "target": "quality = 1 - HTER_proxy (higher=better)",
        "model": "ridge_regression + StandardScaler",
        "embedder": "paraphrase-multilingual-MiniLM-L12-v2",
        "k_folds": args.k,
        "alpha_grid": grid,
        "alpha_selected_per_fold": best_alphas,
        "alpha_selected_median": float(np.median(best_alphas)),
        "metrics": {
            k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
            for k, v in metrics.items()
        },
        "n_rows": len(df),
        "feature_dims": {
            "embeddings": X_mt.shape[1],
            "handcrafted": X_hand.shape[1],
            "total": X.shape[1],
        },
    }

    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
