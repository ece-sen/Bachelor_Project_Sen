from rank_bm25 import BM25Okapi
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.selector.schema_repr import load_schemas, load_queries


class LexicalSelector:
    """
    BM25-based database selector.

    BM25 (Best Match 25) is a ranking function used in information retrieval.
    It scores how relevant a document (schema) is to a query based on:
    - Term frequency: how often query words appear in the schema
    - Inverse document frequency: how rare those words are across all schemas
    - Document length normalization: longer schemas aren't unfairly favored
    """

    def __init__(self, schemas: dict):
        """
        Builds the BM25 index at initialization time.
        This happens once — then you can query it as many times as you want.

        schemas: dict of { db_id -> schema text }
        """
        self.db_ids = list(schemas.keys())

        # BM25 expects a list of tokenized documents
        # Each document is a list of words
        tokenized_schemas = [
            schemas[db_id].lower().split()
            for db_id in self.db_ids
        ]

        self.bm25 = BM25Okapi(tokenized_schemas)

    def score(self, query: str) -> dict:
        """
        Returns a raw BM25 score for every database.
        { db_id -> float score }
        """
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        return {db_id: float(score) for db_id, score in zip(self.db_ids, scores)}

    def rank(self, query: str, top_k: int = 3) -> list:
        """
        Returns top_k databases sorted by BM25 score descending.
        [ (db_id, score), (db_id, score), ... ]
        """
        scores = self.score(query)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

if __name__ == "__main__":
    from src.evaluation.metrics import evaluate

    schemas = load_schemas("data/spider/database")
    queries = load_queries("data/spider/dev.json")

    selector = LexicalSelector(schemas)

    # Sanity check on first 5 queries
    print("=== BM25 Lexical Selector — Sanity Check ===\n")
    for q in queries[:5]:
        question = q["question"]
        correct_db = q["db_id"]
        ranked = selector.rank(question, top_k=3)
        is_correct = ranked[0][0] == correct_db
        print(f"Question:     {question}")
        print(f"Correct DB:   {correct_db}")
        print(f"Top 3 ranked: {ranked}")
        print(f"Result:       {'correct' if is_correct else 'wrong'}")
        print()

    # Full evaluation
    print("=== Full Dev Set Evaluation ===\n")
    top1, top3 = evaluate(selector, queries)
    print(f"Total queries : {len(queries)}")
    print(f"Top-1 Accuracy: {top1:.3f}")
    print(f"Top-3 Accuracy: {top3:.3f}")