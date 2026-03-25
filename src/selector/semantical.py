from sentence_transformers import SentenceTransformer, util
import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.selector.schema_repr import load_schemas, load_queries, Preprocessor
from src.evaluation.metrics import evaluate

# E5 Models
E5_MODELS = {
    "intfloat/e5-small-v2",
    "intfloat/e5-base-v2",
    "intfloat/e5-large-v2",
}

# Nomic Models
NOMIC_MODELS = {
    "nomic-ai/nomic-embed-text-v1",
    "nomic-ai/nomic-embed-text-v1.5",
}

class SemanticSelector:

    def __init__(self, schemas: dict, model_name: str):
        self.db_ids     = list(schemas.keys())
        self.model_name = model_name
        self.is_e5      = model_name in E5_MODELS
        self.is_nomic   = model_name in NOMIC_MODELS

        print(f"Loading {model_name}...")
        self._init_dense(schemas)
        print("Schema encoding complete.\n")

    def _init_dense(self, schemas: dict):
        """
        Initializes a dense embedding model (SBERT, BGE, E5, Nomic, GTE).
        Handles prefix requirements per model family.
        """
        trust_remote = self.is_nomic  # nomic needs trust_remote_code

        self.model = SentenceTransformer(
            self.model_name,
            trust_remote_code=trust_remote
        )

        schema_texts = [schemas[db_id] for db_id in self.db_ids]

        # apply document-side prefix if needed
        if self.is_e5:
            schema_texts = ["passage: " + t for t in schema_texts]
        elif self.is_nomic:
            schema_texts = ["search_document: " + t for t in schema_texts]

        print(f"Encoding {len(self.db_ids)} schemas...")
        self.schema_embeddings = self.model.encode(
            schema_texts,
            convert_to_tensor=True,
            show_progress_bar=True
        )


    def score(self, query: str) -> dict:
        """Scores all databases against the query using cosine similarity."""
        return self._score_dense(query)

    def _score_dense(self, query: str) -> dict:
        """Cosine similarity between query and schema embeddings."""
        # apply query-side prefix if needed
        if self.is_e5:
            query = "query: " + query
        elif self.is_nomic:
            query = "search_query: " + query

        query_embedding = self.model.encode(
            query,
            convert_to_tensor=True
        )
        similarities = util.cos_sim(
            query_embedding, self.schema_embeddings
        )[0]

        return {
            db_id: float(similarities[i])
            for i, db_id in enumerate(self.db_ids)
        }

    def rank(self, query: str, top_k: int = 3) -> list:
        scores = self.score(query)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


if __name__ == "__main__":
    raw_schemas = load_schemas("data/spider/database")
    queries     = load_queries("data/spider/dev.json")

    models_to_compare = [
        "all-MiniLM-L6-v2",
        "BAAI/bge-small-en-v1.5",
        "BAAI/bge-base-en-v1.5",
        "all-mpnet-base-v2",  
        # current best
        "thenlper/gte-small",
        # new additions
        "intfloat/e5-base-v2",
        "intfloat/e5-small-v2",
        "nomic-ai/nomic-embed-text-v1",              
    ]

    print(f"\n=== Extended Semantic Model Comparison ===")
    print(f"{'Model':<60} {'Top-1':>7} {'Top-3':>7} {'MRR':>7}")
    print(f"{'-'*82}")

    results = []
    for model_name in models_to_compare:
        try:
            print(f"\nLoading {model_name}...")
            selector = SemanticSelector(raw_schemas, model_name=model_name)
            r        = evaluate(selector, queries)
            results.append((model_name, r))
            print(f"{model_name:<60} {r['top1']:>7.3f} "
                  f"{r['top3']:>7.3f} {r['mrr']:>7.3f}")
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

    # sorted summary
    results.sort(key=lambda x: x[1]["top1"], reverse=True)
    print(f"\n{'='*82}")
    print("Ranked by Top-1:")
    print(f"{'='*82}")
    print(f"{'Model':<60} {'Top-1':>7} {'Top-3':>7} {'MRR':>7}")
    print(f"{'-'*82}")
    for model_name, r in results:
        print(f"{model_name:<60} {r['top1']:>7.3f} "
              f"{r['top3']:>7.3f} {r['mrr']:>7.3f}")