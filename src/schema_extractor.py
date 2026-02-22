import sqlite3

def extract_schema(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    schema_text = ""

    for table in tables:
        table_name = table[0]
        schema_text += f"Table: {table_name}\n"

        cursor.execute(f'PRAGMA table_info("{table_name}");')        
        columns = cursor.fetchall()
        for column in columns:
            col_name = column[1]
            col_type = column[2]
            schema_text += f"  - {col_name} ({col_type})\n"

        schema_text += "\n"

    conn.close()
    return schema_text