from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.selector.schema_repr import load_schemas, load_queries, Preprocessor
from src.evaluation.metrics import evaluate
from src.selector.lexical import LexicalSelector


class TFIDFSelector:
    """
    TF-IDF based database selector.
    
    TF (Term Frequency): how often a word appears in a schema.
    IDF (Inverse Document Frequency): how rare that word is across
    ALL schemas. A word like 'id' appears in every database schema,
    so it gets a very low IDF score and contributes almost nothing
    to similarity. A word like 'singer' appears in very few schemas,
    so it gets a high IDF score and contributes a lot.

    This is smarter than BM25 for your use case because:
    - It down-weights generic column names like 'id', 'name', 'date'
      that appear across many schemas and are useless for selection
    - It up-weights domain-specific terms that only appear in a
      few schemas, making them strong selection signals

    However TF-IDF still cannot handle:
    - Pluralization (singers ≠ singer)
    - Synonyms (stations ≠ stadiums)
    This is why SBERT needs to be used after this.

    Similarity is measured using cosine similarity, it measures
    the angle between two vectors in TF-IDF space, which is more
    meaningful than raw dot product because it is length-normalized.
    
    UPDATE:
    Accepts a Preprocessor instance which is applied to queries
    at score time, matching whatever preprocessing was applied
    to schemas during load_schemas.
    """

    def __init__(self, schemas: dict, preprocessor: Preprocessor = None, ngram_range: tuple = (1, 1)):
        """
        Fits the TF-IDF vectorizer on all schemas at initialization.
        This builds the vocabulary and computes IDF weights once.

        schemas: dict of { db_id -> schema text }
        ngram_range: (1,1) unigrams only, (1,2) adds bigrams, (1,3) adds trigrams
        preprocessor: applied to queries at score time
        """
        self.db_ids = list(schemas.keys())
        self.preprocessor = preprocessor
        schema_texts = [schemas[db_id] for db_id in self.db_ids]

        # TfidfVectorizer handles tokenization, lowercasing internally
        # sublinear_tf=True applies log normalization to term frequency
        # which prevents very long schemas from dominating
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            sublinear_tf=True, 
            ngram_range=ngram_range
        )

        # Fit on all schemas and transform them into TF-IDF vectors
        # schema_matrix shape: (166, vocabulary_size)
        self.schema_matrix = self.vectorizer.fit_transform(schema_texts)

    def score(self, query: str) -> dict:
        if self.preprocessor:
            query = self.preprocessor.process(query)
        """
        Returns cosine similarity score between the query and every schema.
        { db_id -> float score }

        The query is transformed using the same vocabulary and IDF weights
        that were learned from the schemas during __init__.
        """
        # Transform query into the same TF-IDF vector space
        query_vector = self.vectorizer.transform([query])

        # Compute cosine similarity between query and all schemas
        # Result shape: (1, 166) — one score per schema
        similarities = cosine_similarity(query_vector, self.schema_matrix)[0]

        return {db_id: float(score) for db_id, score
                in zip(self.db_ids, similarities)}

    def rank(self, query: str, top_k: int = 3) -> list:
        """
        Returns top_k databases sorted by TF-IDF cosine similarity descending.
        [ (db_id, score), (db_id, score), ... ]
        """
        scores = self.score(query)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


if __name__ == "__main__": 

    preprocessor = Preprocessor(remove_generic=True, lemmatize=True)
    schemas = load_schemas("data/spider/database", preprocessor=preprocessor)
    queries = load_queries("data/spider/dev.json")

    print("Running TF-IDF...")
    tfidf_selector = TFIDFSelector(schemas, preprocessor=preprocessor)

    # Sanity check on first 5 queries
    print(f"=== TF-IDF Selector with preprocessor values {preprocessor.remove_generic, preprocessor.lemmatize} — Sanity Check ===\n")
    for q in queries[:5]:
        question = q["question"]
        correct_db = q["db_id"]
        ranked = tfidf_selector.rank(question, top_k=3)
        is_correct = ranked[0][0] == correct_db
        print(f"Question:     {question}")
        print(f"Correct DB:   {correct_db}")
        print(f"Top 3 ranked: {ranked}")
        print(f"Result:       {'correct' if is_correct else 'wrong'}")
        print()

    # Full evaluation
    print("=== Full Dev Set Evaluation ===\n")
    r = evaluate(tfidf_selector, queries)
    print(f"Total queries : {len(queries)}")
    print(f"Top-1 Accuracy: {r['top1']:.3f}")
    print(f"Top-3 Accuracy: {r['top3']:.3f}")
    print(f"MRR           : {r['mrr']:.3f}")