from sentence_transformers import SentenceTransformer, util
import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.selector.schema_repr import load_schemas, load_queries, Preprocessor
from src.evaluation.metrics import evaluate


class SemanticSelector:
    """
    SBERT-based database selector.

    BM25 and TF-IDF both failed on cases like:
        query:  "stations"  → schema has: "stadium"
        query:  "nation"    → schema has: "country"
        query:  "French"    → schema has: "France"

    These failures happen because both methods compare exact tokens.
    They have zero understanding of meaning.

    SBERT solves this by converting text into dense vectors called
    embeddings. An embedding is a list of 384 numbers that encodes
    the meaning of a piece of text. Texts with similar meanings
    produce vectors that point in similar directions in 384-dimensional
    space, even if they share zero words.

    For example:
        "singers"  → [0.23, -0.11, 0.87, ...]
        "singer"   → [0.24, -0.10, 0.86, ...]  ← very close
        "nation"   → [0.31, -0.05, 0.79, ...]
        "country"  → [0.30, -0.06, 0.80, ...]  ← close
        "airport"  → [-0.45, 0.33, -0.12, ...] ← far from singer

    Similarity between two vectors is measured with cosine similarity —
    the cosine of the angle between them:
        cos = 1.0  → identical direction → identical meaning
        cos = 0.0  → perpendicular → unrelated
        cos = -1.0 → opposite direction → opposite meaning

    In practice for schema matching you see scores between 0.1 and 0.6.

    Model choice — all-MiniLM-L6-v2:
        - 22MB, fast on CPU
        - 384-dimensional embeddings
        - trained on 1 billion sentence pairs
        - good balance of speed and accuracy
        - first run downloads model to ~/.cache/huggingface

    Important: SBERT should NOT receive preprocessed text.
    Stopword removal and lemmatization destroy sentence structure
    that the transformer model needs to understand meaning.
    "How many singers do we have?" is meaningful to SBERT.
    "many singer" is not.
    """

    def __init__(self, schemas: dict,
                 model_name: str = "all-MiniLM-L6-v2"):
        """
        Loads the SBERT model and pre-computes embeddings for all schemas.

        Pre-computing schema embeddings at startup is critical.
        Encoding one text takes ~1ms on CPU. With 166 schemas you
        pay this cost once at startup (~0.2s) instead of on every
        single query. At query time you only encode the query itself.

        schemas:    dict of { db_id -> schema text }
                    should be loaded WITHOUT preprocessing since SBERT
                    works better on natural text
        model_name: which SBERT model to use
        """
        self.db_ids = list(schemas.keys())
        self.model = SentenceTransformer(model_name)

        print(f"Encoding {len(self.db_ids)} schemas with {model_name}...")

        schema_texts = [schemas[db_id] for db_id in self.db_ids]

        # encode() converts each schema text into a 384-dim vector
        # shape of result: (166, 384)
        # convert_to_tensor=True keeps them as torch tensors which is
        # faster for cosine similarity computation than numpy arrays
        self.schema_embeddings = self.model.encode(
            schema_texts,
            convert_to_tensor=True,
            show_progress_bar=True
        )

        print("Schema encoding complete.\n")

    def score(self, query: str) -> dict:
        """
        Encodes the query and computes cosine similarity against
        all pre-computed schema embeddings.

        Returns { db_id -> float similarity score }

        This is called once per query at evaluation time.
        Schema embeddings are already computed — only the query
        needs to be encoded here, which takes ~1ms.
        """
        # encode query into a 384-dim vector
        # shape: (384,)
        query_embedding = self.model.encode(
            query,
            convert_to_tensor=True
        )

        # util.cos_sim computes cosine similarity between query vector
        # and every schema vector simultaneously using matrix operations
        # result shape: (1, 166) — one score per schema
        similarities = util.cos_sim(query_embedding,
                                    self.schema_embeddings)[0]

        return {
            db_id: float(similarities[i])
            for i, db_id in enumerate(self.db_ids)
        }

    def rank(self, query: str, top_k: int = 3) -> list:
        """
        Returns top_k databases sorted by semantic similarity descending.
        [ (db_id, score), (db_id, score), ... ]
        """
        scores = self.score(query)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


if __name__ == "__main__":
    from src.selector.lexical import LexicalSelector
    from src.selector.statistical import TFIDFSelector

    # SBERT works on raw schema text — no preprocessing
    # BM25 and TF-IDF use best configuration from experimental.py
    raw_schemas = load_schemas("data/spider/database")
    best_preprocessor = Preprocessor(remove_generic=True, lemmatize=True)
    best_schemas = load_schemas("data/spider/database",
                                preprocessor=best_preprocessor)
    queries = load_queries("data/spider/dev.json")

    # Sanity check on first 5 queries
    print("=== SBERT Semantic Selector — Sanity Check ===\n")
    sbert_selector = SemanticSelector(raw_schemas)

    for q in queries[:5]:
        question = q["question"]
        correct_db = q["db_id"]
        ranked = sbert_selector.rank(question, top_k=3)
        is_correct = ranked[0][0] == correct_db
        print(f"Question:     {question}")
        print(f"Correct DB:   {correct_db}")
        print(f"Top 3 ranked: {ranked}")
        print(f"Result:       {'correct' if is_correct else 'wrong'}")
        print()

    # Full evaluation
    print("=== Full Dev Set Evaluation ===\n")
    sbert_top1, sbert_top3 = evaluate(sbert_selector, queries)

    # Compare all methods
    print("Running BM25 best config for comparison...")
    bm25_top1, bm25_top3 = evaluate(
        LexicalSelector(best_schemas, preprocessor=best_preprocessor),
        queries
    )

    print("Running TF-IDF best config for comparison...")
    tfidf_top1, tfidf_top3 = evaluate(
        TFIDFSelector(best_schemas, preprocessor=best_preprocessor,
                      ngram_range=(1, 2)),
        queries
    )

    # Full comparison table
    print("\n=== Comparison: All Methods ===")
    print(f"{'Method':<30} {'Top-1':>8} {'Top-3':>8}")
    print(f"{'-'*48}")
    print(f"{'BM25 (stop+lemma)':<30} {bm25_top1:>8.3f} {bm25_top3:>8.3f}")
    print(f"{'TF-IDF (stop+lemma+bi)':<30} {tfidf_top1:>8.3f} {tfidf_top3:>8.3f}")
    print(f"{'SBERT (raw)':<30} {sbert_top1:>8.3f} {sbert_top3:>8.3f}")

    # Show cases where SBERT succeeds but TF-IDF fails
    # The examples for H2
    print("\n=== Cases where SBERT wins over TF-IDF ===\n")
    tfidf_selector = TFIDFSelector(best_schemas,
                                   preprocessor=best_preprocessor,
                                   ngram_range=(1, 2))
    wins = 0
    for q in queries:
        question = q["question"]
        correct_db = q["db_id"]

        tfidf_ranked = tfidf_selector.rank(question, top_k=1)
        sbert_ranked = sbert_selector.rank(question, top_k=1)

        tfidf_wrong = tfidf_ranked[0][0] != correct_db
        sbert_right = sbert_ranked[0][0] == correct_db

        # SBERT got it right where TF-IDF failed
        if tfidf_wrong and sbert_right and wins < 5:
            print(f"  Query:         {question}")
            print(f"  Correct DB:    {correct_db}")
            print(f"  TF-IDF said:   {tfidf_ranked[0][0]}")
            print(f"  SBERT said:    {sbert_ranked[0][0]}")
            print()
            wins += 1

    models_to_compare = [
    "all-MiniLM-L6-v2",        # baseline SBERT
    "all-mpnet-base-v2",        # stronger SBERT
    "BAAI/bge-small-en-v1.5",  # small BGE
    "BAAI/bge-base-en-v1.5",   # medium BGE
    ]

    print(f"\n=== Embedding Model Comparison ===")
    print(f"{'Model':<35} {'Top-1':>8} {'Top-3':>8}")
    print(f"{'-'*53}")

    for model_name in models_to_compare:
        print(f"Loading {model_name}...")
        selector = SemanticSelector(raw_schemas, model_name=model_name)
        top1, top3 = evaluate(selector, queries)
        print(f"{model_name:<35} {top1:>8.3f} {top3:>8.3f}")
