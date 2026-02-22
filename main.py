from src.schema_extractor import extract_schema

northwind_schema = extract_schema("Databases/sakila.db")
print("NORTHWIND SCHEMA:")
print(northwind_schema[:500])  # print first 500 chars
