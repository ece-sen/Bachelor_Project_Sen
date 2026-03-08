import sqlite3
import re
from pathlib import Path

# Try removing generic words, stopwords
STOPWORDS =  {
    "id", "name", "date", "time", "type", "code", "status",
    "description", "comment", "notes", "created", "updated",
    "the", "a", "an", "and", "or", "but", "in", "on", "at",
    "to", "for", "of", "with", "by"
}

class Preprocessor:
    """
    Handles all text preprocessing before similarity scoring.

    A single Preprocessor instance is created once and passed to
    every selector. This guarantees schemas and queries are always
    processed identically.

    Parameters:
        remove_generic: removes stopwords from text
        lemmatize:      reduces words to base form (singers → singer)

    N-grams are not handled here — they are a vectorizer parameter
    inside TFIDFSelector, not a text transformation.
    """

    def __init__(self, remove_generic: bool = False, lemmatize: bool = False):
        self.remove_generic = remove_generic
        self.lemmatize = lemmatize

        # Only load lemmatizer if needed
        self._lemmatizer = None
        if lemmatize:
            from nltk.stem import WordNetLemmatizer
            self._lemmatizer = WordNetLemmatizer()

    def process(self, text: str) -> str:
        """
        Applies all configured preprocessing steps to any text.
        Called on schema texts at load time and on queries at score time.
        Same method, same result — no mismatch possible.
        """
        text = text.lower()

        if self.remove_generic:
            tokens = text.split()
            tokens = [t for t in tokens if t not in STOPWORDS]
            text = " ".join(tokens)

        if self.lemmatize and self._lemmatizer:
            tokens = text.split()
            tokens = [self._lemmatizer.lemmatize(t) for t in tokens]
            text = " ".join(tokens)

        return text

    def __repr__(self):
        """
        Shows preprocessor settings clearly in print outputs.
        Useful for labeling experiment results.
        """
        parts = []
        if self.remove_generic:
            parts.append("stopwords")
        if self.lemmatize:
            parts.append("lemmatize")
        return f"Preprocessor({', '.join(parts) if parts else 'baseline'})"


def extract_schema_from_db(db_path: str,  preprocessor: Preprocessor = None) -> str:
    """
    Extracts schema text directly from a SQLite file.
    Queries SQLite's internal metadata via PRAGMA table_info.

    Applies preprocessor if provided.

    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    parts = []

    # Add the database name (folder name) 
    db_name = Path(db_path).parent.name
    parts.append(db_name.replace("_", " "))

    # Get all table names from SQLite's internal master table
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]

    for table in tables:
        # Add table name
        parts.append(table.replace("_", " "))

        # Get all column names for this table
        cursor.execute(f"PRAGMA table_info('{table}')")
        columns = cursor.fetchall()
        # PRAGMA table_info returns: (cid, name, type, notnull, dflt_value, pk)
        # index 1 is the column name
        for col in columns:
            parts.append(col[1].replace("_", " "))

    conn.close()
    text = " ".join(parts)

    if preprocessor:
        text = preprocessor.process(text)

    return text


def load_schemas(database_dir: str, preprocessor: Preprocessor = None) -> dict:
    """
    Walks the Spider database directory and builds schema text
    for every SQLite file found.
    Returns dict: { db_id -> schema text }
    """
    schemas = {}
    db_root = Path(database_dir)

    for db_folder in db_root.iterdir():
        if db_folder.is_dir():
            sqlite_files = list(db_folder.glob("*.sqlite"))
            if sqlite_files:
                db_id = db_folder.name
                schemas[db_id] = extract_schema_from_db(str(sqlite_files[0]), preprocessor=preprocessor)

    return schemas


def load_queries(queries_path: str) -> list:
    """
    Loads queries from any Spider split (dev.json, train_spider.json).
    Each entry has 'question' and 'db_id'.
    """
    import json
    with open(queries_path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_words(text: str) -> set:
    """Lowercase, remove punctuation, split into words."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', '', text)
    return set(text.split())


if __name__ == "__main__":
   
    schemas = load_schemas("data/spider/database")
    queries = load_queries("data/spider/dev.json")

    print(f"Loaded {len(schemas)} database schemas")
    print(f"Loaded {len(queries)} queries\n")

    example_q = queries[0]
    correct_db = example_q["db_id"]

    print(f"Example question: {example_q['question']}")
    print(f"Ground truth DB:  {correct_db}")
    print(f"Schema text:      {schemas[correct_db][:300]}\n")

    print("--- Vocabulary divergence examples ---")
    for q in queries[:45]:
        db_id = q["db_id"]
        schema_words = clean_words(schemas[db_id])
        query_words = clean_words(q["question"])
        overlap = query_words & schema_words
        divergence = (query_words - schema_words) - STOPWORDS
        if len(divergence) > 1:
            print(f"Query:      {q['question']}")
            print(f"DB:         {db_id}")
            print(f"Overlap:    {overlap - STOPWORDS}")
            print(f"No match:   {divergence}")
            print()