"""
RQ2 — Hybrid Database Selector

Combines BM25, TF-IDF, and a semantic (SBERT/E5/GTE) selector
into a single score using weighted linear fusion:

    hybrid_score(q, db) = α·bm25 + β·tfidf + γ·semantic

All three raw score dicts are min-max normalized to [0, 1]
before fusion so the different score ranges (BM25 is unbounded,
cosine similarity is already [0, 1]) don't skew the result.

If a selector is not provided its weight is redistributed
proportionally to the remaining selectors, so you can also
run this as a two-selector hybrid.
"""

import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.selector.schema_repr import load_schemas, load_queries, Preprocessor
from src.selector.lexical import LexicalSelector
from src.selector.statistical import TFIDFSelector
from src.selector.semantical import SemanticSelector
from src.evaluation.metrics import evaluate
import json
from pathlib import Path

def _minmax_normalize(scores: dict) -> dict:
  
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi == lo:
        return {db_id: 0.0 for db_id in scores}
    return {db_id: (v - lo) / (hi - lo) for db_id, v in scores.items()}


class HybridSelector:

    def __init__(
        self,
        bm25_selector=None,
        tfidf_selector=None,
        semantic_selector=None,
        weights: tuple = (0.1, 0.2, 0.7),
    ):
        selectors_weights = [
            (bm25_selector,     weights[0]),
            (tfidf_selector,    weights[1]),
            (semantic_selector, weights[2]),
        ]

        # drop None selectors and re-normalize remaining weights
        active = [(s, w) for s, w in selectors_weights if s is not None]
        if len(active) < 2:
            raise ValueError("HybridSelector needs at least two selectors.")

        total_weight = sum(w for _, w in active)
        self._selectors = [(s, w / total_weight) for s, w in active]

    def score(self, query: str) -> dict:
        
        fused = None

        for selector, weight in self._selectors:
            raw    = selector.score(query)
            normed = _minmax_normalize(raw)

            if fused is None:
                fused = {db_id: weight * v for db_id, v in normed.items()}
            else:
                for db_id, v in normed.items():
                    fused[db_id] += weight * v

        return fused

    def rank(self, query: str, top_k: int = 3) -> list:
        scores = self.score(query)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


def grid_search(
    bm25_selector,
    tfidf_selector,
    semantic_selector,
    queries: list,
    step: float = 0.1,
) -> tuple:
   
    best_top1   = -1.0
    best_weights = None
    all_results  = []

    # generate all (a, b, c) with a+b+c == 1.0 at given step size
    vals = np.arange(0.0, 1.0 + step, step)
    vals = [round(float(v), 2) for v in vals]

    for alpha in vals:
        for beta in vals:
            gamma = round(1.0 - alpha - beta, 2)
            if gamma < 0.0 or gamma > 1.0:
                continue

            selector = HybridSelector(
                bm25_selector=bm25_selector,
                tfidf_selector=tfidf_selector,
                semantic_selector=semantic_selector,
                weights=(alpha, beta, gamma),
            )
            r = evaluate(selector, queries)

            all_results.append((alpha, beta, gamma, r["top1"], r["top3"], r["mrr@3"], r["mrr@10"]))

            if r["top1"] > best_top1:
                best_top1    = r["top1"]
                best_weights = (alpha, beta, gamma)

    # sort and print top 10
    all_results.sort(key=lambda x: x[3], reverse=True)
    print(f"\n{'Alpha':>7} {'Beta':>7} {'Gamma':>7} "
          f"{'Top-1':>8} {'Top-3':>8} {'MRR@3':>8} {'MRR@10':>8}")
    print("-" * 82)
    for row in all_results[:10]:
        a, b, g, t1, t3, mrr_3, mrr_10 = row
        print(f"{a:>7.2f} {b:>7.2f} {g:>7.2f} "
              f"{t1:>8.3f} {t3:>8.3f} {mrr_3:>8.3f} {mrr_10:>8.3f}")

    print(f"\nBest weights → alpha={best_weights[0]}, "
          f"beta={best_weights[1]}, gamma={best_weights[2]}  "
          f"Top-1={best_top1:.3f}")

    return best_weights


if __name__ == "__main__":
    queries = load_queries("data/spider/dev.json")

    # Best preprocessing from for lexical/statistical selectors
    p = Preprocessor(remove_generic=True, lemmatize=True)
    schemas_preprocessed = load_schemas("data/spider/database", preprocessor=p)

    # Semantic selectors always use raw schemas
    raw_schemas = load_schemas("data/spider/database")

    print("Initializing selectors...")
    bm25   = LexicalSelector(schemas_preprocessed, preprocessor=p, variant="okapi")
    tfidf  = TFIDFSelector(schemas_preprocessed, preprocessor=p, ngram_range=(1, 2))
    sbert  = SemanticSelector(raw_schemas, model_name="thenlper/gte-small")

    # individual baselines for reference
    print("\n=== Individual selector baselines ===")
    print(f"{'Selector':<20} {'Top-1':>8} {'Top-3':>8} {'MRR@3':>8} {'MRR@10':>8}")
    print("-" * 82)
    for label, sel in [("BM25", bm25), ("TF-IDF", tfidf), ("GTE-small", sbert)]:
        r = evaluate(sel, queries)
        print(f"{label:<20} {r['top1']:>8.3f} {r['top3']:>8.3f} {r['mrr@3']:>8.3f} {r['mrr@10']:>8.3f}")

    # Fixed-weight hybrid first
    print("\n=== Fixed-weight hybrid (0.1 / 0.2 / 0.7) ===")
    hybrid = HybridSelector(bm25, tfidf, sbert, weights=(0.1, 0.2, 0.7))
    r = evaluate(hybrid, queries)
    print(f"{'Hybrid':<20} {r['top1']:>8.3f} {r['top3']:>8.3f} {r['mrr@3']:>8.3f} {r['mrr@10']:>8.3f}")

    # Grid search for optimal weights
    print("\n=== Grid search (step=0.1) — top 10 weight combos ===")
    best = grid_search(bm25, tfidf, sbert, queries, step=0.1)

    # Evaluate best found weights
    print(f"\n=== Best hybrid (α={best[0]}, β={best[1]}, γ={best[2]}) ===")
    best_hybrid = HybridSelector(bm25, tfidf, sbert, weights=best)
    r = evaluate(best_hybrid, queries)
    print(f"{'Best Hybrid':<20} {r['top1']:>8.3f} {r['top3']:>8.3f} {r['mrr@3']:>8.3f} {r['mrr@10']:>8.3f}")

    # Two-selector ablations
    print("\n=== Two-selector ablations ===")
    print(f"{'Combination':<30} {'Top-1':>8} {'Top-3':>8} {'MRR@3':>8} {'MRR@10':>8}")
    print("-" * 82)

    combos = [
        ("BM25 + TF-IDF",    bm25,  tfidf, None,  (0.4, 0.6, 0.0)),
        ("BM25 + Semantic",  bm25,  None,  sbert, (0.3, 0.0, 0.7)),
        ("TF-IDF + Semantic", None, tfidf, sbert, (0.0, 0.3, 0.7)),
    ]
    for label, b, t, s, w in combos:
        h = HybridSelector(b, t, s, weights=w)
        r = evaluate(h, queries)
        print(f"{label:<30} {r['top1']:>8.3f} {r['top3']:>8.3f} {r['mrr@3']:>8.3f} {r['mrr@10']:>8.3f}")

    # Save best weights to disk for final evaluation on test set 
    Path("models").mkdir(exist_ok=True)
    with open("models/best_hybrid_weights.json", "w") as f:
        json.dump({"weights": list(best)}, f)
    print(f"\nBest weights saved to models/best_hybrid_weights.json")