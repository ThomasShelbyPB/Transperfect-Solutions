import pandas as pd
import numpy as np
import re
from difflib import SequenceMatcher
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr, spearmanr


# ----------------------------
# Data loading
# ----------------------------
def read_ch2_xlsx(path):
    df = pd.read_excel(path)

    df = df.rename(columns={
        "English Source": "src",
        "MT System": "mt",
        "Post-Edit Text": "pe",
        "Nature of change/Comments": "comment"
    })

    for c in ["src", "mt", "pe", "comment"]:
        df[c] = df[c].fillna("").astype(str)

    return df


# ----------------------------
# HTER proxy (edit distance)
# ----------------------------
def hter_proxy(mt: str, pe: str) -> float:
    sm = SequenceMatcher(None, mt, pe)
    return 1.0 - sm.ratio()


def compute_quality_scores(df):
    df["hter"] = df.apply(lambda r: hter_proxy(r["mt"], r["pe"]), axis=1)
    df["quality"] = 1.0 - df["hter"]  # higher = better
    return df


# ----------------------------
# Feature helpers
# ----------------------------
def _words(s: str):
    return re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+", s.lower())


def _coverage(src: str, mt: str):
    sw = set(_words(src))
    mw = set(_words(mt))
    if not sw:
        return 0.0, 0.0
    overlap = sw & mw
    recall = len(overlap) / max(len(sw), 1)      # missing content proxy
    precision = len(overlap) / max(len(mw), 1) if mw else 0.0
    return recall, precision


def _digit_mismatch(src: str, mt: str) -> float:
    return float(re.findall(r"\d+", src) != re.findall(r"\d+", mt))


def _tag_ok(src: str, mt: str) -> float:
    return float(src.count("[TAG]") == mt.count("[TAG]"))


def handcrafted_features(src: str, mt: str):
    cov_r, cov_p = _coverage(src, mt)
    return np.array([
        abs(len(src) - len(mt)) / max(len(src), 1),   # length diff
        len(mt) / max(len(src), 1),                    # length ratio
        SequenceMatcher(None, src, mt).ratio(),        # similarity
        cov_r,                                        # content recall
        cov_p,                                        # content precision
        _digit_mismatch(src, mt),                      # numbers wrong?
        _tag_ok(src, mt),                              # TAG placement ok?
        mt.count("."),
        mt.count(","),
        mt.count(":"),
        mt.count("!"),
        mt.count("?"),
        float(src.isupper() != mt.isupper()),          # casing issue
    ], dtype=np.float32)


def regression_metrics(y_true, y_pred):
    return {
        "pearson": pearsonr(y_true, y_pred)[0],
        "spearman": spearmanr(y_true, y_pred)[0],
        "mae": mean_absolute_error(y_true, y_pred),
    }
