"""
Ablation study for RQ1:
Effect of stopword removal, lemmatization, and n-grams
on BM25 and TF-IDF accuracy.

Each experiment isolates one variable so results are comparable.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.selector.schema_repr import load_schemas, load_queries, Preprocessor
from src.selector.lexical import LexicalSelector
from src.selector.statistical import TFIDFSelector
from src.evaluation.metrics import evaluate


def run_experiment(label: str, queries: list,
                   preprocessor: Preprocessor,
                   ngram_range: tuple = (1, 1)):
    """
    Loads schemas with given preprocessor, runs BM25 and TF-IDF,
    prints one result row per method.

    The same preprocessor is passed to both load_schemas and the
    selectors.
    """
    schemas = load_schemas("data/spider/database", preprocessor=preprocessor)

    bm_r = evaluate(
        LexicalSelector(schemas, preprocessor=preprocessor, variant="okapi"), queries)

    tfidf_r = evaluate(
        TFIDFSelector(schemas, preprocessor=preprocessor,
                      ngram_range=ngram_range), queries)

    print(f"{'BM25':<10} {label:<35} {bm_r['top1']:>8.3f} {bm_r['top3']:>8.3f}")
    print(f"{'TF-IDF':<10} {label:<35} {tfidf_r['top1']:>8.3f} {tfidf_r['top3']:>8.3f}")


if __name__ == "__main__":
    queries = load_queries("data/spider/dev.json")

    print(f"{'Method':<10} {'Configuration':<35} {'Top-1':>8} {'Top-3':>8}")
    print("-" * 63)

    # Experiment 1 — baseline
    run_experiment("baseline", queries,
                   Preprocessor())

    # Experiment 2 — stopwords only
    run_experiment("stopwords", queries,
                   Preprocessor(remove_generic=True))

    # Experiment 3 — lemmatization only
    run_experiment("lemmatization", queries,
                   Preprocessor(lemmatize=True))

    # Experiment 4 — stopwords + lemmatization
    run_experiment("stopwords + lemmatization", queries,
                   Preprocessor(remove_generic=True, lemmatize=True))

    # Experiment 5 — bigrams only
    # BM25 stays unigram, only TF-IDF uses bigrams
    run_experiment("bigrams (TF-IDF only)", queries,
                   Preprocessor(), ngram_range=(1, 2))

    # Experiment 6 — best combination
    run_experiment("stopwords + lemma + bigrams (TF-IDF only)", queries,
                   Preprocessor(remove_generic=True, lemmatize=True),
                   ngram_range=(1, 2))