import sqlite3
import re
from pathlib import Path


def extract_schema_from_db(db_path: str) -> str:
    """
    Extracts schema text directly from a SQLite file.
    Queries SQLite's internal metadata via PRAGMA table_info.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    parts = []

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
    return " ".join(parts)


def load_schemas(database_dir: str) -> dict:
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
                schemas[db_id] = extract_schema_from_db(str(sqlite_files[0]))

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
        divergence = query_words - schema_words
        if len(divergence) > 2:
            print(f"Query:      {q['question']}")
            print(f"DB:         {db_id}")
            print(f"Overlap:    {overlap}")
            print(f"No match:   {divergence}")
            print()