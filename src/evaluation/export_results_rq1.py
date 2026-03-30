"""
Exports per-query results for each method to JSON files.
Each JSON file contains full details for every query:
    - schema texts used by that method
    - original and preprocessed query
    - word overlap
    - top-1 and top-3 predictions
    - correctness flag

Output: results/feature_results/method_name.json
"""
import sys
import os
import json
import re
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.selector.schema_repr import (
    load_schemas, load_queries, Preprocessor, clean_words, STOPWORDS
)
from src.selector.lexical import LexicalSelector
from src.selector.statistical import TFIDFSelector
from src.selector.semantical import SemanticSelector
from src.evaluation.metrics import compute_mrr, evaluate


def get_overlap(query_text: str, schema_text: str) -> list:
    """
    Returns list of words that appear in both the
    processed query and the schema text.
    Excludes stopwords so only meaningful matches are shown.
    """
    def tokenize(text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9 ]', '', text)
        return set(text.split())

    query_words  = tokenize(query_text)
    schema_words = tokenize(schema_text)
    overlap = query_words & schema_words
    # remove stopwords from overlap so only content words shown
    overlap = {w for w in overlap if w not in STOPWORDS}
    return sorted(list(overlap))


def export_method(
    method_name:  str,
    selector,
    schemas:      dict,
    queries:      list,
    preprocessor: Preprocessor = None
):
   
    os.makedirs("results/feature_results", exist_ok=True)

    output = {
        "method": method_name,
        "total_queries": len(queries),
        "schemas": {
            db_id: schemas[db_id]
            for db_id in sorted(schemas.keys())
        },
        "queries": []
    }

    top1_correct = 0
    top3_correct = 0
    mrr3_sum = 0.0
    mrr10_sum = 0.0

    for q in queries:
        original_question = q["question"]
        correct_db        = q["db_id"]

        # apply same preprocessing to query that was applied to schemas
        if preprocessor:
            query_used = preprocessor.process(original_question)
        else:
            query_used = original_question

        # get top 3 predictions from selector
        ranked_3   = selector.rank(original_question, top_k=3)
        top1_db  = ranked_3[0][0]
        top3_with_scores = [
            {"db_id": db, "score": round(score, 4)}
            for db, score in ranked_3
        ]

        # get MRR@10 scores for this query
        ranked_10  = selector.rank(original_question, top_k=10)
        top10_with_scores = [
            {"db_id": db, "score": round(score, 4)}
            for db, score in ranked_10
        ]

        # compute word overlap between processed query and correct schema
        overlap = get_overlap(query_used, schemas[correct_db])

        is_correct = top1_db == correct_db
        if is_correct:
            top1_correct += 1
        if correct_db in [t["db_id"] for t in top3_with_scores]: 
            top3_correct += 1

        mrr3_sum  += compute_mrr([t["db_id"] for t in top3_with_scores],  correct_db)
        mrr10_sum += compute_mrr([t["db_id"] for t in top10_with_scores], correct_db)

        output["queries"].append({
            "original_question": original_question,
            "query_used":        query_used,
            "correct_db":        correct_db,
            "correct_db_schema": schemas[correct_db],
            "overlap_with_correct_schema": overlap,
            "top1_db":           top1_db,
            "top3":              top3_with_scores,
            "top10":             top10_with_scores,
            "top1_correct":      is_correct
        })

    # summary stats at the top level for quick reference
    output["top1_accuracy"] = round(top1_correct / len(queries), 4)
    output["top1_correct_count"] = top1_correct
    output["top3_accuracy"] = round(top3_correct / len(queries), 4)
    output["top3_correct_count"] = top3_correct
    output["mrr@3"] = round(mrr3_sum / len(queries), 4)
    output["mrr@10"] = round(mrr10_sum / len(queries), 4)

    filepath = f"results/feature_results/{method_name.replace(' ', '_')}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"  Exported {filepath}  "
          f"(Top-1: {output['top1_accuracy']:.3f}  "
          f"MRR@3: {output['mrr@3']:.3f}  "
          f"MRR@10: {output['mrr@10']:.3f}  "
          f"{top1_correct}/{len(queries)} correct)")


if __name__ == "__main__":
    queries = load_queries("data/spider/dev.json")

    print("Exporting results for all methods...\n")

    # ── BM25 baseline ──────────────────────────────────────────
    p = Preprocessor()
    schemas = load_schemas("data/spider/database", preprocessor=p)
    export_method(
        "BM25_baseline",
        LexicalSelector(schemas, preprocessor=p, variant="okapi"),
        schemas, queries, preprocessor=p
    )

    # ── BM25 stopwords ──────────────────────────────────────────
    p = Preprocessor(remove_generic=True, lemmatize=False)
    schemas = load_schemas("data/spider/database", preprocessor=p)
    export_method(
        "BM25_stopwords",
        LexicalSelector(schemas, preprocessor=p, variant="okapi"),
        schemas, queries, preprocessor=p
    )

    # ── BM25 lemmatization ──────────────────────────────────────────
    p = Preprocessor(remove_generic=False, lemmatize=True)
    schemas = load_schemas("data/spider/database", preprocessor=p)
    export_method(
        "BM25_lemmatization",
        LexicalSelector(schemas, preprocessor=p, variant="okapi"),
        schemas, queries, preprocessor=p
    )

    # ── BM25 stopwords + lemmatization ─────────────────────────
    p = Preprocessor(remove_generic=True, lemmatize=True)
    schemas = load_schemas("data/spider/database", preprocessor=p)
    export_method(
        "BM25_stop_lemma",
        LexicalSelector(schemas, preprocessor=p, variant="okapi"),
        schemas, queries, preprocessor=p
    )

     # ── BM25Plus — best preprocessing ──────────────────────────
    p = Preprocessor(remove_generic=True, lemmatize=True)
    schemas = load_schemas("data/spider/database", preprocessor=p)
    export_method(
        "BM25Plus_stop_lemma",
        LexicalSelector(schemas, preprocessor=p, variant="plus"),
        schemas, queries, preprocessor=p
    )

    # ── BM25L — best preprocessing ─────────────────────────────
    p = Preprocessor(remove_generic=True, lemmatize=True)
    schemas = load_schemas("data/spider/database", preprocessor=p)
    export_method(
        "BM25L_stop_lemma",
        LexicalSelector(schemas, preprocessor=p, variant="l"),
        schemas, queries, preprocessor=p
    )

    # ── TF-IDF baseline ────────────────────────────────────────
    p = Preprocessor()
    schemas = load_schemas("data/spider/database", preprocessor=p)
    export_method(
        "TFIDF_baseline",
        TFIDFSelector(schemas, preprocessor=p),
        schemas, queries, preprocessor=p
    )

    # ── TF-IDF stopwords ───────────────────────
    p = Preprocessor(remove_generic=True, lemmatize=False)
    schemas = load_schemas("data/spider/database", preprocessor=p)
    export_method(
        "TFIDF_stopwords",
        TFIDFSelector(schemas, preprocessor=p),
        schemas, queries, preprocessor=p
    )

    # ── TF-IDF lemmatization ───────────────────────
    p = Preprocessor(remove_generic=False, lemmatize=True)
    schemas = load_schemas("data/spider/database", preprocessor=p)
    export_method(
        "TFIDF_lemmatization",
        TFIDFSelector(schemas, preprocessor=p),
        schemas, queries, preprocessor=p
    )

    # ── TF-IDF stopwords + lemmatization ───────────────────────
    p = Preprocessor(remove_generic=True, lemmatize=True)
    schemas = load_schemas("data/spider/database", preprocessor=p)
    export_method(
        "TFIDF_stop_lemma",
        TFIDFSelector(schemas, preprocessor=p),
        schemas, queries, preprocessor=p
    )

    # ── TF-IDF best — stopwords + lemma + bigrams ──────────────
    p = Preprocessor(remove_generic=True, lemmatize=True)
    schemas = load_schemas("data/spider/database", preprocessor=p)
    export_method(
        "TFIDF_stop_lemma_bigrams",
        TFIDFSelector(schemas, preprocessor=p, ngram_range=(1, 2)),
        schemas, queries, preprocessor=p
    )

    # ── SBERT MiniLM — no preprocessing, raw schemas ───────────
    # SBERT works on natural text, never preprocessed
    raw_schemas = load_schemas("data/spider/database")

    export_method(
        "SBERT_MiniLM",
        SemanticSelector(raw_schemas, model_name="all-MiniLM-L6-v2"),
        raw_schemas, queries, preprocessor=None
    )

    # ── SBERT MPNet ─────────────────────────────────────────────
    export_method(
        "SBERT_MPNet",
        SemanticSelector(raw_schemas, model_name="all-mpnet-base-v2"),
        raw_schemas, queries, preprocessor=None
    )

    # ── BGE small ───────────────────────────────────────────────
    export_method(
        "BGE_small",
        SemanticSelector(raw_schemas, model_name="BAAI/bge-small-en-v1.5"),
        raw_schemas, queries, preprocessor=None
    )

    # ── BGE base ────────────────────────────────────────────────
    export_method(
        "BGE_base",
        SemanticSelector(raw_schemas, model_name="BAAI/bge-base-en-v1.5"),
        raw_schemas, queries, preprocessor=None
    )

    # ── GTE small ───────────────────────────────────────────────
    export_method(
        "GTE_small",
        SemanticSelector(raw_schemas,
                         model_name="thenlper/gte-small"),
        raw_schemas, queries, preprocessor=None
    )

    # ── E5 base ─────────────────────────────────────────────────
    export_method(
        "E5_base",
        SemanticSelector(raw_schemas,
                         model_name="intfloat/e5-base-v2"),
        raw_schemas, queries, preprocessor=None
    )

    # ── E5 small ─────────────────────────────────────────────────
    export_method(
        "E5_small",
        SemanticSelector(raw_schemas,
                         model_name="intfloat/e5-small-v2"),
        raw_schemas, queries, preprocessor=None
    )

    # ── Nomic embed v1 ──────────────────────────────────────────
    export_method(
        "Nomic_embed_v1",
        SemanticSelector(raw_schemas,
                         model_name="nomic-ai/nomic-embed-text-v1"),
        raw_schemas, queries, preprocessor=None
    )

    print("\nDone. JSON files saved in results/feature_results/")