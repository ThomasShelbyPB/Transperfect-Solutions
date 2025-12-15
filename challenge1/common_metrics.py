import re
from collections import Counter
from typing import List, Tuple

def tokenize_words(s: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", str(s), re.UNICODE)

def levenshtein(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[m]

def hter_proxy(mt: str, ref: str) -> float:
    mt_t = tokenize_words(mt)
    ref_t = tokenize_words(ref)
    dist = levenshtein(mt_t, ref_t)
    return dist / max(1, len(ref_t))

def chrf_proxy(hyp: str, ref: str, n: int = 6, beta: float = 2.0) -> float:
    """Simplified chrF (character n-gram F-score), averaged n=1..n."""
    hyp = str(hyp)
    ref = str(ref)

    def ngrams(s: str, k: int):
        return [s[i:i+k] for i in range(len(s)-k+1)] if len(s) >= k else []

    scores = []
    for k in range(1, n + 1):
        hyp_ng = ngrams(hyp, k)
        ref_ng = ngrams(ref, k)
        if not hyp_ng or not ref_ng:
            continue
        c_h = Counter(hyp_ng)
        c_r = Counter(ref_ng)
        overlap = sum(min(c_h[g], c_r[g]) for g in c_h.keys())
        prec = overlap / max(1, sum(c_h.values()))
        rec = overlap / max(1, sum(c_r.values()))
        if prec + rec == 0:
            f = 0.0
        else:
            f = (1 + beta**2) * prec * rec / (beta**2 * prec + rec)
        scores.append(f)
    return sum(scores) / len(scores) if scores else 0.0
