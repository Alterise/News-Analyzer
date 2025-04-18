import os
import psycopg2
from psycopg2 import sql

# Database configuration
DB_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB"),
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD"),
    "host": os.getenv("POSTGRES_HOST"),
    "port": os.getenv("POSTGRES_PORT", "5432"),
}

def execute_query_from_file(file_path):
    """Execute SQL query from a file"""
    try:
        # Read the query from file
        with open(file_path, 'r') as file:
            query = file.read()
        
        if not query.strip():
            print("Error: Query file is empty")
            return
        
        # Connect to the database
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Execute the query
        cursor.execute(query)
        
        # If it's a SELECT query, fetch and print results
        if query.strip().lower().startswith('select'):
            results = cursor.fetchall()
            print("Query executed successfully. Results:")
            for row in results:
                print(row)
        else:
            conn.commit()
            print("Query executed successfully. Rows affected:", cursor.rowcount)
            
    except Exception as e:
        print("Error executing query:", e)
        if 'conn' in locals():
            conn.rollback()
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    query_file = "query.txt"
    if not os.path.exists(query_file):
        print(f"Error: File '{query_file}' not found")
    else:
        execute_query_from_file(query_file)