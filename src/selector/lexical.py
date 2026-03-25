from rank_bm25 import BM25Okapi, BM25Plus, BM25L
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.selector.schema_repr import load_schemas, load_queries, Preprocessor
from src.evaluation.metrics import evaluate



class LexicalSelector:
    """
    BM25-based database selector.

    BM25 (Best Match 25) is a ranking function used in information retrieval.
    It scores how relevant a document (schema) is to a query based on:
    - Term frequency: how often query words appear in the schema
    - Inverse document frequency: how rare those words are across all schemas
    - Document length normalization: longer schemas aren't unfairly favored
    """

    def __init__(self, schemas: dict, preprocessor: Preprocessor = None, variant: str = "okapi"):
        """
        Builds the BM25 index at initialization time.
        This happens once — then it can be queried as many times as want.

        schemas: dict of { db_id -> schema text }
        """
        self.db_ids = list(schemas.keys())
        self.preprocessor = preprocessor

        tokenized_schemas = []
        for db_id in self.db_ids:
            text = schemas[db_id]
            # schemas are already preprocessed by load_schemas
            # so here we just tokenize
            tokenized_schemas.append(text.lower().split())
            
        if variant == "plus":
            self.bm25 = BM25Plus(tokenized_schemas)
        elif variant == "l":
            self.bm25 = BM25L(tokenized_schemas)
        else:
            self.bm25 = BM25Okapi(tokenized_schemas)

    def score(self, query: str) -> dict:
        # apply same preprocessing to query as was applied to schemas
        if self.preprocessor:
            query = self.preprocessor.process(query)
        else:
            query = query.lower()
        tokens = query.split()
        scores = self.bm25.get_scores(tokens)
        return {db_id: float(score) for db_id, score in zip(self.db_ids, scores)}

    def rank(self, query: str, top_k: int = 3) -> list:
        scores = self.score(query)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

if __name__ == "__main__":

    preprocessor = Preprocessor()
    schemas = load_schemas("data/spider/database", preprocessor=preprocessor)
    queries = load_queries("data/spider/dev.json")

    selector = LexicalSelector(schemas, preprocessor=preprocessor)

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
    r = evaluate(selector, queries)
    print(f"Total queries : {len(queries)}")
    print(f"Top-1 Accuracy: {r['top1']:.3f}")
    print(f"Top-3 Accuracy: {r['top3']:.3f}")
    print(f"MRR           : {r['mrr']:.3f}")