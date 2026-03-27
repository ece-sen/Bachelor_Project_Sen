import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.selector.schema_repr import load_schemas, load_queries, Preprocessor
from src.selector.lexical     import LexicalSelector
from src.selector.statistical import TFIDFSelector
from src.selector.semantical  import SemanticSelector
from src.evaluation.metrics   import evaluate
from mlp_fusion               import FusionMLP, MLPFusionSelector

DATA_TEST_PATH    = "data/spider/test.json"         # questions + db_id labels
DATABASE_DIR      = "data/spider/database"          # train/dev schemas (for selectors)
TEST_DATABASE_DIR = "data/spider/test_database"     # test schemas (for evaluation)

CKPT_FINETUNED = "models/gte-small-finetuned"
CKPT_HARDNEG   = "models/gte-small-hardneg"
CKPT_MLP       = "models/mlp_fusion.pt"
BASE_MODEL     = "thenlper/gte-small"


def check_checkpoints():
   
    missing = []
    for path in [CKPT_FINETUNED, CKPT_HARDNEG, CKPT_MLP]:
        if not Path(path).exists():
            missing.append(path)
    if missing:
        print("WARNING — missing checkpoints (run train_all.py first):")
        for m in missing:
            print(f"  {m}")
        print()


def run_eval(label, selector, test_qs):
   
    r = evaluate(selector, test_qs)
    return {"label": label, **r}


def print_table(rows, n_queries):
    W = 62
    print("\n" + "═" * W)
    print(f"Final Evaluation — Spider Test Set  (N={n_queries} queries)")
    print("═" * W)
    print(f"{'Model':<38} {'Top-1':>6} {'Top-3':>6} {'MRR':>6}")
    print("─" * W)
    for r in rows:
        print(f"{r['label']:<38} {r['top1']:>6.3f} "
              f"{r['top3']:>6.3f} {r['mrr']:>6.3f}")
    print("─" * W)
    best = max(rows, key=lambda x: x["top1"])
    print(f"\nBest: {best['label']}   "
          f"Top-1={best['top1']:.3f}   MRR={best['mrr']:.3f}")
    print("═" * W)


if __name__ == "__main__":

    check_checkpoints()

    # Load data
    print("Loading schemas and test queries...")
    p                    = Preprocessor(remove_generic=True, lemmatize=True)

    # train/dev schemas — used to build BM25/TF-IDF/semantic selectors
    schemas_preprocessed = load_schemas(DATABASE_DIR, preprocessor=p)
    raw_schemas          = load_schemas(DATABASE_DIR)

    # test schemas — used to score against during evaluation
    test_schemas_preprocessed = load_schemas(TEST_DATABASE_DIR, preprocessor=p)
    test_raw_schemas          = load_schemas(TEST_DATABASE_DIR)

    test_qs = load_queries(DATA_TEST_PATH)

    print(f"  Test queries   : {len(test_qs)}")
    print(f"  Test schemas   : {len(test_raw_schemas)}")
    print(f"  Train schemas  : {len(raw_schemas)}")

    # Shared base selectors built on TEST schemas
    print("\nInitializing base selectors on test schemas...")
    bm25     = LexicalSelector(test_schemas_preprocessed, preprocessor=p,
                               variant="okapi")
    tfidf    = TFIDFSelector(test_schemas_preprocessed, preprocessor=p,
                             ngram_range=(1, 2))

    results = []

    # Model 1: GTE-small base
    print("\nEvaluating Model 1 — GTE-small base...")
    semantic_base = SemanticSelector(test_raw_schemas, model_name=BASE_MODEL)
    r1 = evaluate(semantic_base, test_qs[:5])
    if r1["top1"] == 0.0:
        print("  WARNING: Model 1 scoring 0.0 on first 5 queries — check test file format")
        print(f"  Sample query: {test_qs[0]['question']}")
        print(f"  Sample db_id: {test_qs[0]['db_id']}")
        print(f"  Top-3 result: {semantic_base.rank(test_qs[0]['question'], top_k=3)}")
    results.append(run_eval("GTE-small base", semantic_base, test_qs))

    # Model 2: GTE-small fine-tuned (random negatives)
    if Path(CKPT_FINETUNED).exists():
        print("Evaluating Model 2 — GTE-small fine-tuned (random neg)...")
        semantic_ft = SemanticSelector(test_raw_schemas, model_name=CKPT_FINETUNED)
        results.append(run_eval("GTE-small fine-tuned (random neg)",
                                semantic_ft, test_qs))
    else:
        print(f"Skipping Model 2 — checkpoint not found: {CKPT_FINETUNED}")

    # Model 3: GTE-small fine-tuned (hard negatives)
    if Path(CKPT_HARDNEG).exists():
        print("Evaluating Model 3 — GTE-small fine-tuned (hard neg)...")
        semantic_hn = SemanticSelector(test_raw_schemas, model_name=CKPT_HARDNEG)
        results.append(run_eval("GTE-small fine-tuned (hard neg)",
                                semantic_hn, test_qs))
    else:
        print(f"Skipping Model 3 — checkpoint not found: {CKPT_HARDNEG}")

    # Model 4: MLP fusion
    if Path(CKPT_MLP).exists():
        print("Evaluating Model 4 — MLP fusion (BM25+TF-IDF+GTE)...")
        mlp_selector = MLPFusionSelector(bm25, tfidf, semantic_base, CKPT_MLP)
        results.append(run_eval("MLP fusion (BM25+TF-IDF+GTE)",
                                mlp_selector, test_qs))
    else:
        print(f"Skipping Model 4 — checkpoint not found: {CKPT_MLP}")

    # Also include fixed-weight hybrid from hybrid.py as a reference
    from hybrid import HybridSelector
    import json
    weights_path = Path("models/best_hybrid_weights.json")
    if weights_path.exists():
        best_w = tuple(json.load(open(weights_path))["weights"])
        print(f"Evaluating best hybrid (α={best_w[0]}, β={best_w[1]}, γ={best_w[2]})...")
        label = f"Best hybrid ({best_w[0]}/{best_w[1]}/{best_w[2]})"
    else:
        best_w = (0.0, 0.2, 0.8)   # known best from grid search / will be dynamically imported from json later
        print(f"Evaluating best hybrid (α=0.0, β=0.2, γ=0.8)...")
        label = "Best hybrid (0.0/0.2/0.8)"
    hybrid = HybridSelector(bm25, tfidf, semantic_base, weights=best_w)
    results.append(run_eval(label, hybrid, test_qs))

    # Print final table
    results.sort(key=lambda x: x["top1"], reverse=True)
    print_table(results, len(test_qs))