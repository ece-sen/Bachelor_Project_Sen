"""
pipeline_utils.py

Shared utilities for the NL2SQL pipeline notebooks.
Handles:
  - Full schema extraction (CREATE TABLE format) for the NL2SQL prompt
  - SQL execution against SQLite databases
  - Execution accuracy evaluation
"""

import sqlite3
import os
import re
from pathlib import Path


# ── Schema extraction ──────────────────────────────────────────────────────

def get_create_table_schema(db_path: str) -> str:
    """
    Extracts the full CREATE TABLE schema from a SQLite file.
    This is what the NL2SQL model needs — not the flat token string
    used for selection, but the proper SQL schema with column types.

    Example output:
        CREATE TABLE singer (
            Singer_ID INTEGER PRIMARY KEY,
            Name TEXT,
            Country TEXT,
            ...
        );
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]

    schema_parts = []
    for table in tables:
        cursor.execute(
            f"SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
            (table,)
        )
        row = cursor.fetchone()
        if row and row[0]:
            schema_parts.append(row[0].strip() + ";")

    conn.close()
    return "\n\n".join(schema_parts)


def get_db_path(database_dir: str, db_id: str) -> str:
    """Returns the path to the SQLite file for a given db_id."""
    db_folder = os.path.join(database_dir, db_id)
    sqlite_files = list(Path(db_folder).glob("*.sqlite"))
    if not sqlite_files:
        raise FileNotFoundError(f"No SQLite file found for db_id: {db_id}")
    return str(sqlite_files[0])


# ── SQL execution ──────────────────────────────────────────────────────────

def execute_sql(db_path: str, sql: str):
    """
    Executes a SQL query against a SQLite database.
    Returns the result rows as a sorted list of tuples,
    or None if execution fails.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()
        # Sort for order-independent comparison
        return sorted([tuple(str(v).lower() for v in row) for row in results])
    except Exception:
        return None


def execution_match(db_path: str, pred_sql: str, gold_sql: str) -> bool:
    """
    Returns True if predicted SQL produces the same result as gold SQL.
    This is the standard Spider execution accuracy metric.
    """
    pred_result = execute_sql(db_path, pred_sql)
    gold_result = execute_sql(db_path, gold_sql)

    if pred_result is None or gold_result is None:
        return False
    return pred_result == gold_result


# ── SQL cleaning ───────────────────────────────────────────────────────────

def clean_generated_sql(raw_output: str) -> str:
    """
    Cleans the raw output from the NL2SQL model.
    SQLCoder sometimes wraps output in markdown code blocks
    or includes explanation text — this strips all of that.
    """
    # Remove markdown code blocks
    raw_output = re.sub(r'```sql', '', raw_output, flags=re.IGNORECASE)
    raw_output = re.sub(r'```', '', raw_output)

    # Take only the first SQL statement (up to first semicolon or end)
    lines = raw_output.strip().split('\n')
    sql_lines = []
    for line in lines:
        # Stop if we hit an explanation line after the SQL
        if sql_lines and line.strip() and not line.strip().startswith('--'):
            if any(line.upper().strip().startswith(kw)
                   for kw in ('NOTE', 'EXPLANATION', 'THIS', 'THE', 'HERE')):
                break
        sql_lines.append(line)

    sql = ' '.join(sql_lines).strip()

    # Remove trailing semicolons for consistency
    sql = sql.rstrip(';').strip()

    return sql


# ── Prompt builder ─────────────────────────────────────────────────────────

def build_sqlcoder_prompt(question: str, schema: str, db_id: str) -> str:
    """
    Builds the prompt in SQLCoder's expected format.
    See: https://huggingface.co/defog/sqlcoder-7b-2
    """
    return f"""### Task
Generate a SQL query to answer the following question:
`{question}`

### Database Schema
The query will run on a database with the following schema:
{schema}

### Answer
Given the database schema, here is the SQL query that answers `{question}`:
```sql
"""


# ── Results logger ─────────────────────────────────────────────────────────

def make_result_record(
    question:    str,
    db_id:       str,
    selected_db: str,
    gold_sql:    str,
    pred_sql:    str,
    is_correct:  bool,
    db_correct:  bool,
) -> dict:
    """Creates a standardized result record for logging."""
    return {
        "question":        question,
        "correct_db":      db_id,
        "selected_db":     selected_db,
        "db_selection_correct": db_correct,
        "gold_sql":        gold_sql,
        "predicted_sql":   pred_sql,
        "execution_match": is_correct,
    }
