"""
export_model_results.py
 
Exports per-query results for the RQ2 models to JSON files:
    - Best hybrid  (α=0.0, β=0.2, γ=0.8  — TF-IDF + GTE-small)
    - MLP fusion   (BM25 + TF-IDF + GTE-small)
 
Each JSON file contains full details for every query:
    - original question
    - correct database
    - top-1 and top-3 predictions with scores
    - individual component scores (bm25, tfidf, semantic) — normalized
    - correctness flag
 
Output: results/model_results/<model_name>.json
 
Usage:
    python export_model_results.py
 
    # To export only one model:
    python export_model_results.py --model hybrid
    python export_model_results.py --model mlp
"""

# Import standard libraries and set up paths

import sys
import os
import json
import sys
import os
import json
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.selector.schema_repr import load_schemas, load_queries, Preprocessor
from src.selector.lexical     import LexicalSelector
from src.selector.statistical import TFIDFSelector
from src.selector.semantical  import SemanticSelector
from src.evaluation.metrics   import compute_mrr
from training_models.hybrid                   import HybridSelector, _minmax_normalize
from training_models.mlp_fusion               import MLPFusionSelector
from training_models.train_all import BASE_MODEL

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

TEST_DATABASE_DIR   = os.path.join(_ROOT, "data", "spider", "test_database")
TEST_PATH           = os.path.join(_ROOT, "data", "spider", "test.json")
MLP_PATH            = os.path.join(_ROOT, "models", "mlp_fusion.pt")
HYBRID_WEIGHTS_PATH = os.path.join(_ROOT, "models", "best_hybrid_weights.json")
OUTPUT_DIR          = os.path.join(_ROOT, "results", "model_results")


def load_hybrid_weights(path) -> tuple:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        weights = tuple(data["weights"])
        print(f"Loaded hybrid weights from {path}: {weights}")
        return weights
    else:
        print(f"Hybrid weights file not found at {path}. Using default (0.0, 0.2, 0.8).")
        return (0.0, 0.2, 0.8)
    

def component_scores(question, bm25, tfidf, semantic) -> dict:
    '''Computes normalized component scores for a question.'''
    bm25_norm     = _minmax_normalize(bm25.score(question))
    tfidf_norm    = _minmax_normalize(tfidf.score(question))
    semantic_norm = _minmax_normalize(semantic.score(question))
    all_dbs = list(bm25_norm.keys())
    return {
        db: {
            "bm25":     round(bm25_norm[db], 4),
            "tfidf":    round(tfidf_norm[db], 4),
            "semantic": round(semantic_norm[db], 4),
        }
        for db in all_dbs
    }

def export_model(model_name: str, selector, queries: list, bm25, tfidf, semantic):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output = { "model": model_name, "total_queries": len(queries), "queries": [] }

    top1_correct = 0
    top3_correct = 0
    mrr3_total = 0.0
    mrr10_total = 0.0

    for i, q in enumerate(queries):
        if i % 100 == 0:
            print(f" {i}/{len(queries)} queries processed...")
        question = q["question"]
        correct_db = q["db_id"]

        # ranked predictions
        ranked_3 = selector.rank(question, top_k=3)
        ranked_10 = selector.rank(question, top_k=10)

        top1_db = ranked_3[0][0]

        top3 = [{"db_id": db, "score": round(score, 4)} for db, score in ranked_3]
        top10 = [{"db_id": db, "score": round(score, 4)} for db, score in ranked_10]

        # correctness
        is_top1_correct = (top1_db == correct_db)
        is_top3_correct = correct_db in [t["db_id"] for t in top3]

        if is_top1_correct:
            top1_correct += 1
        if is_top3_correct:
            top3_correct += 1

        mrr3_total += compute_mrr([t["db_id"] for t in top3], correct_db)
        mrr10_total += compute_mrr([t["db_id"] for t in top10], correct_db)

        # component scores for top-3 candidates
        relevant_dbs = list({correct_db} | {t["db_id"] for t in top3})
        comp_scores = component_scores(question, bm25, tfidf, semantic)
        relevant_component_scores = {db: comp_scores[db] for db in relevant_dbs if db in comp_scores}

        # rank of correct DB in top10
        rank_of_correct = next((idx + 1 for idx, t in enumerate(top10) if t["db_id"] == correct_db), None)

        output["queries"].append({
            "question": question,
            "correct_db": correct_db,
            "top1": top1_db,
            "top3": top3,
            "top10": top10,
            "is_top1_correct": is_top1_correct,
            "is_top3_correct": is_top3_correct,
            "mrr3_contribution": compute_mrr([t["db_id"] for t in top3], correct_db),
            "mrr10_contribution": compute_mrr([t["db_id"] for t in top10], correct_db),
            "component_scores": relevant_component_scores,
            "rank_of_correct_in_top10": rank_of_correct,
        })

    n = len(queries)
    output["metrics"] = {
        "top1_accuracy": round(top1_correct / n, 4),
        "top3_accuracy": round(top3_correct / n, 4),
        "mrr@3": round(mrr3_total / n, 4),
        "mrr@10": round(mrr10_total / n, 4),
        "top1_correct_count": top1_correct,
        "top3_correct_count": top3_correct,
    }

    output_path = os.path.join(OUTPUT_DIR, f"{model_name}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Exported results for {model_name} to {output_path}")

# main
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["hybrid", "mlp", "all"], default="all",
                        help="Which model(s) to export results for")
    args = parser.parse_args()

    print("Loading schemas and test queries...")
    p = Preprocessor(remove_generic=True, lemmatize=True)
    test_schemas_preprocessed = load_schemas(TEST_DATABASE_DIR, preprocessor=p)
    test_raw_schemas          = load_schemas(TEST_DATABASE_DIR)
    test_qs                   = load_queries(TEST_PATH)

    print(f"  {len(test_qs)} queries | {len(test_raw_schemas)} databases\n")

    print("Initializing base selectors...")
    bm25     = LexicalSelector(test_schemas_preprocessed, preprocessor=p, variant="okapi")
    tfidf    = TFIDFSelector(test_schemas_preprocessed, preprocessor=p, ngram_range=(1, 2))
    semantic = SemanticSelector(test_raw_schemas, model_name=BASE_MODEL)

    if args.model in ("hybrid", "all"):
        print("Exporting results for Hybrid Selector...")
        hybrid_weights = load_hybrid_weights(HYBRID_WEIGHTS_PATH)
        hybrid_selector = HybridSelector(bm25, tfidf, semantic, weights=hybrid_weights)
        export_model("best_weights_hybrid", hybrid_selector, test_qs, bm25, tfidf, semantic)

    if args.model in ("mlp", "all"):
        print("Exporting results for MLP Fusion Selector...")
        mlp_selector = MLPFusionSelector(bm25, tfidf, semantic, MLP_PATH)
        export_model("mlp_fusion", mlp_selector, test_qs, bm25, tfidf, semantic)

    print("\nAll done. Results exported to JSON files in the results/model_results/ directory.")