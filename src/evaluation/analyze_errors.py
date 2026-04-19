"""
analyze_errors.py

Loads per-query JSON exported by export_model_results.py and
produces a structured error analysis of all Top-1 failures.

For each failure, assigns one or more error categories:

    NEAR_MISS         -- correct DB was ranked 2nd or 3rd
    FAR_MISS          -- correct DB not in top-10
    SCHEMA_CONFUSION  -- predicted and correct DB share many schema tokens
    VOCAB_DIVERGENCE  -- query words barely appear in the correct schema
    AMBIGUOUS         -- query is very short/generic

Results saved to:
    results/model_results/error_analysis_<model>.json  -- full per-query details
    results/model_results/error_summary_<model>.txt    -- human-readable summary

Usage:
    python analyze_errors.py --model mlp_fusion
    python analyze_errors.py --model best_weights_hybrid
    python analyze_errors.py --model all
"""

import sys
import os
import json
import re
import argparse
from collections import Counter

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(_ROOT)

from src.selector.schema_repr import load_schemas, STOPWORDS

# paths
DATABASE_DIR = os.path.join(_ROOT, "data", "spider", "test_database")
INPUT_DIR    = os.path.join(_ROOT, "results", "model_results")
OUTPUT_DIR   = os.path.join(_ROOT, "results", "model_results")

# thresholds
SCHEMA_CONFUSION_THRESHOLD = 0.4   # Jaccard similarity between schema token sets
VOCAB_DIVERGENCE_THRESHOLD = 0.10  # fraction of query words found in correct schema
AMBIGUOUS_MAX_TOKENS       = 6     # queries with <= this many content words


# helpers

def tokenize(text: str) -> set:
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', '', text)
    tokens = set(text.split())
    return tokens - STOPWORDS


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def overlap_fraction(query_tokens: set, schema_text: str) -> float:
    """Fraction of query content words that appear in the schema."""
    schema_tokens = tokenize(schema_text)
    if not query_tokens:
        return 0.0
    return len(query_tokens & schema_tokens) / len(query_tokens)


def categorize_failure(record: dict, schemas: dict) -> dict:
    """
    Takes one failed query record from the exported JSON and returns
    a dict with error categories and supporting evidence.

    Field names match export_model_results.py exactly:
        record["top1"]                    -- top-1 predicted db_id
        record["rank_of_correct_in_top10"] -- int or None
        record["top3"]                    -- list of {"db_id": ..., "score": ...}
        record["component_scores"]        -- {"db_id": {"bm25", "tfidf", "semantic"}}
    """
    question     = record["question"]
    correct_db   = record["correct_db"]
    predicted_db = record["top1"]
    rank         = record["rank_of_correct_in_top10"]  # None means not in top-10
    top3_dbs     = [t["db_id"] for t in record["top3"]]

    query_tokens   = tokenize(question)
    correct_schema = schemas.get(correct_db, "")
    pred_schema    = schemas.get(predicted_db, "")
    correct_tokens = tokenize(correct_schema)
    pred_tokens    = tokenize(pred_schema)

    categories = []
    evidence   = {}

    # NEAR_MISS
    if rank in (2, 3):
        categories.append("NEAR_MISS")
        evidence["rank_of_correct"] = rank

    # FAR_MISS
    if rank is None:
        categories.append("FAR_MISS")

    # SCHEMA_CONFUSION
    schema_sim = jaccard(correct_tokens, pred_tokens)
    if schema_sim >= SCHEMA_CONFUSION_THRESHOLD:
        categories.append("SCHEMA_CONFUSION")
        evidence["schema_jaccard_correct_vs_pred"] = round(schema_sim, 3)
        evidence["shared_schema_tokens"] = sorted(correct_tokens & pred_tokens)[:20]

    # VOCAB_DIVERGENCE
    overlap_frac = overlap_fraction(query_tokens, correct_schema)
    if overlap_frac <= VOCAB_DIVERGENCE_THRESHOLD:
        categories.append("VOCAB_DIVERGENCE")
        evidence["query_overlap_with_correct_schema"] = round(overlap_frac, 3)
        evidence["query_words_not_in_correct_schema"] = sorted(
            query_tokens - correct_tokens
        )

    # AMBIGUOUS
    if len(query_tokens) <= AMBIGUOUS_MAX_TOKENS:
        categories.append("AMBIGUOUS")
        evidence["query_token_count"] = len(query_tokens)
        evidence["query_tokens"] = sorted(query_tokens)

    # component score evidence
    comp = record.get("component_scores", {})
    if correct_db in comp and predicted_db in comp:
        evidence["scores_correct_db"]   = comp[correct_db]
        evidence["scores_predicted_db"] = comp[predicted_db]
        misleading = [
            sig for sig in ("bm25", "tfidf", "semantic")
            if comp[predicted_db].get(sig, 0) > comp[correct_db].get(sig, 0)
        ]
        if misleading:
            evidence["components_that_favored_wrong_db"] = misleading

    if not categories:
        categories.append("OTHER")

    return {
        "question":        question,
        "correct_db":      correct_db,
        "predicted_db":    predicted_db,
        "rank_of_correct": rank,
        "top3_dbs":        top3_dbs,
        "categories":      categories,
        "evidence":        evidence,
    }


# main analysis

def analyze(model_name: str, schemas: dict):
    input_path = os.path.join(INPUT_DIR, f"{model_name}.json")
    if not os.path.exists(input_path):
        print(f"  File not found: {input_path}")
        print("  Run export_model_results.py first.")
        return

    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    metrics = data["metrics"]

    print(f"\n{'='*60}")
    print(f"Error analysis -- {model_name}")
    print(f"  Total queries : {data['total_queries']}")
    print(f"  Top-1 accuracy: {metrics['top1_accuracy']:.3f}")
    print(f"  Top-3 accuracy: {metrics['top3_accuracy']:.3f}")
    print(f"  MRR@3         : {metrics['mrr@3']:.3f}")
    print(f"  MRR@10        : {metrics['mrr@10']:.3f}")
    print(f"{'='*60}\n")

    # filter failures using the correct field name from export_model_results.py
    failures = [q for q in data["queries"] if not q["is_top1_correct"]]
    print(f"  Failures to analyze: {len(failures)}\n")

    analyzed        = []
    category_counts = Counter()
    component_blame = Counter()

    for record in failures:
        result = categorize_failure(record, schemas)
        analyzed.append(result)
        for cat in result["categories"]:
            category_counts[cat] += 1
        for comp in result["evidence"].get("components_that_favored_wrong_db", []):
            component_blame[comp] += 1

    # category summary
    print("  Error category breakdown:")
    print(f"  {'Category':<25} {'Count':>6}  {'% of failures':>14}")
    print(f"  {'-'*50}")
    for cat, count in category_counts.most_common():
        pct = 100 * count / len(failures)
        print(f"  {cat:<25} {count:>6}  {pct:>13.1f}%")

    print(f"\n  Component blame (signal that favored the wrong DB):")
    print(f"  {'Component':<15} {'Count':>6}")
    print(f"  {'-'*22}")
    for comp, count in component_blame.most_common():
        print(f"  {comp:<15} {count:>6}")

    # near miss summary
    near_misses = [a for a in analyzed if "NEAR_MISS" in a["categories"]]
    print(f"\n  Near misses (rank 2 or 3): {len(near_misses)} "
          f"({100*len(near_misses)/len(failures):.1f}% of failures)")

    # most confused DB pairs
    confusion_pairs = Counter()
    for a in analyzed:
        pair = tuple(sorted([a["correct_db"], a["predicted_db"]]))
        confusion_pairs[pair] += 1

    print(f"\n  Top 10 most confused database pairs:")
    print(f"  {'Correct DB':<30} {'Predicted DB':<30} {'Count':>6}")
    print(f"  {'-'*68}")
    for (db1, db2), count in confusion_pairs.most_common(10):
        print(f"  {db1:<30} {db2:<30} {count:>6}")

    # vocab divergence sample
    vocab_div = [a for a in analyzed if "VOCAB_DIVERGENCE" in a["categories"]]
    print(f"\n  Vocabulary divergence failures: {len(vocab_div)}")
    print(f"  Sample (up to 5):")
    for a in vocab_div[:5]:
        print(f"    Q: {a['question']}")
        print(f"    Correct: {a['correct_db']}  |  Predicted: {a['predicted_db']}")
        missing = a["evidence"].get("query_words_not_in_correct_schema", [])
        print(f"    Query words not in correct schema: {missing[:10]}")
        print()

    # schema confusion sample
    schema_conf = [a for a in analyzed if "SCHEMA_CONFUSION" in a["categories"]]
    print(f"  Schema confusion failures: {len(schema_conf)}")
    print(f"  Sample (up to 5):")
    for a in schema_conf[:5]:
        print(f"    Q: {a['question']}")
        print(f"    Correct: {a['correct_db']}  |  Predicted: {a['predicted_db']}")
        sim    = a["evidence"].get("schema_jaccard_correct_vs_pred", 0)
        shared = a["evidence"].get("shared_schema_tokens", [])
        print(f"    Schema Jaccard: {sim:.3f}  |  Shared tokens: {shared[:8]}")
        print()

    # save JSON
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    analysis_output = {
        "model":           model_name,
        "total_queries":   data["total_queries"],
        "total_failures":  len(failures),
        "metrics":         metrics,
        "category_counts": dict(category_counts),
        "component_blame": dict(component_blame),
        "confusion_pairs": [
            {"db1": db1, "db2": db2, "count": count}
            for (db1, db2), count in confusion_pairs.most_common(20)
        ],
        "failures": analyzed,
    }

    json_path = os.path.join(OUTPUT_DIR, f"error_analysis_{model_name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(analysis_output, f, indent=2, ensure_ascii=False)
    print(f"\n  Full analysis saved -> {json_path}")

    # save txt summary
    txt_path = os.path.join(OUTPUT_DIR, f"error_summary_{model_name}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Error Analysis -- {model_name}\n")
        f.write(f"{'='*60}\n")
        f.write(f"Total queries  : {data['total_queries']}\n")
        f.write(f"Top-1 failures : {len(failures)}\n")
        f.write(f"Top-1 accuracy : {metrics['top1_accuracy']:.3f}\n")
        f.write(f"Top-3 accuracy : {metrics['top3_accuracy']:.3f}\n")
        f.write(f"MRR@3          : {metrics['mrr@3']:.3f}\n")
        f.write(f"MRR@10         : {metrics['mrr@10']:.3f}\n\n")

        f.write("Category breakdown:\n")
        for cat, count in category_counts.most_common():
            pct = 100 * count / len(failures)
            f.write(f"  {cat:<25} {count:>5}  ({pct:.1f}%)\n")

        f.write("\nComponent blame:\n")
        for comp, count in component_blame.most_common():
            f.write(f"  {comp:<15} {count:>5}\n")

        f.write("\nTop 10 confused DB pairs:\n")
        for (db1, db2), count in confusion_pairs.most_common(10):
            f.write(f"  {db1:<30} {db2:<30} {count:>4}\n")

        f.write("\n\nAll failures (sorted by category):\n")
        f.write(f"{'='*60}\n")
        for a in sorted(analyzed, key=lambda x: x["categories"][0]):
            f.write(f"\nQ: {a['question']}\n")
            f.write(f"Correct DB : {a['correct_db']}\n")
            f.write(f"Predicted  : {a['predicted_db']}  "
                    f"(rank of correct: {a['rank_of_correct']})\n")
            f.write(f"Categories : {', '.join(a['categories'])}\n")
            ev = a["evidence"]
            if "schema_jaccard_correct_vs_pred" in ev:
                f.write(f"Schema similarity (Jaccard): "
                        f"{ev['schema_jaccard_correct_vs_pred']}\n")
            if "query_overlap_with_correct_schema" in ev:
                f.write(f"Query-schema overlap: "
                        f"{ev['query_overlap_with_correct_schema']}\n")
            if "components_that_favored_wrong_db" in ev:
                f.write(f"Misleading components: "
                        f"{ev['components_that_favored_wrong_db']}\n")

    print(f"  Summary saved      -> {txt_path}")


# entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["mlp_fusion", "best_weights_hybrid", "all"],
        default="all",
        help="Which model's results to analyze (default: all)"
    )
    args = parser.parse_args()

    print("Loading schemas for analysis...")
    raw_schemas = load_schemas(DATABASE_DIR)
    print(f"  {len(raw_schemas)} schemas loaded.\n")

    if args.model in ("mlp_fusion", "all"):
        analyze("mlp_fusion", raw_schemas)

    if args.model in ("best_weights_hybrid", "all"):
        analyze("best_weights_hybrid", raw_schemas)