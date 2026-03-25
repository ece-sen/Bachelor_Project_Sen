"""
Reads all result JSON files from results/feature_results and prints
a unified comparison table sorted by Top-1 accuracy.

No selectors are re-initialized — reads from saved JSON files.
"""
import json
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


def load_result(filepath: str) -> dict:
    """
    Loads one result JSON and computes Top-3 and Top-5
    from the query-level data since older exports only
    stored top1_accuracy at summary level.
    """
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    queries = data["queries"]
    total   = len(queries)

    top1 = sum(1 for q in queries if q["top1_correct"]) / total

    # top3 — check if correct_db is in top3 list
    top3 = sum(
        1 for q in queries
        if q["correct_db"] in [t["db_id"] for t in q["top3"]]
    ) / total

    # mrr — reciprocal rank
    mrr_sum = 0.0
    for q in queries:
        top3_dbs = [t["db_id"] for t in q["top3"]]
        if q["correct_db"] in top3_dbs:
            rank = top3_dbs.index(q["correct_db"]) + 1
            mrr_sum += 1.0 / rank
    mrr = mrr_sum / total

    return {
        "method":  data["method"],
        "top1":    round(top1, 4),
        "top3":    round(top3, 4),
        "mrr":     round(mrr,  4),
        "correct": int(top1 * total),
        "total":   total,
    }


def print_summary_table(results_dir: str = "results/feature_results"):
    """
    Loads all JSON files in results/feature_results and prints
    a unified table sorted by Top-1 accuracy.
    """
    json_files = [
        f for f in os.listdir(results_dir)
        if f.endswith(".json")
    ]

    if not json_files:
        print(f"No JSON files found in {results_dir}/")
        return

    rows = []
    for fname in sorted(json_files):
        filepath = os.path.join(results_dir, fname)
        try:
            row = load_result(filepath)
            rows.append(row)
        except Exception as e:
            print(f"  Could not load {fname}: {e}")

    # sort by Top-1 descending
    rows.sort(key=lambda x: x["top1"], reverse=True)

    # group into method families for visual separation
    def get_family(method_name: str) -> str:
        name = method_name.upper()
        if "BGE"   in name: return "BGE"
        if "E5"    in name: return "E5"
        if "GTE"   in name: return "GTE"
        if "SBERT" in name or "MINILM" in name or "MPNET" in name:
            return "SBERT"
        if "TFIDF" in name: return "TFIDF"
        if "BM25"  in name: return "BM25"
        return "OTHER"

    print("\n" + "=" * 72)
    print("Complete Results — All Methods")
    print("=" * 72)
    print(f"{'Method':<35} {'Top-1':>7} {'Top-3':>7} "
          f"{'MRR':>7} {'Correct':>9}")
    print("-" * 72)

    prev_family = ""
    for row in rows:
        family = get_family(row["method"])

        # print blank line between method families
        if family != prev_family and prev_family != "":
            print()
        prev_family = family

        print(f"{row['method']:<35} "
              f"{row['top1']:>7.3f} "
              f"{row['top3']:>7.3f} "
              f"{row['mrr']:>7.3f} "
              f"{row['correct']:>5}/{row['total']}")

    print("=" * 72)
    print(f"\nBest method: {rows[0]['method']}  "
          f"Top-1={rows[0]['top1']:.3f}  "
          f"MRR={rows[0]['mrr']:.3f}")


if __name__ == "__main__":
    print_summary_table("results/feature_results")