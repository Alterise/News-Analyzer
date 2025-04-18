import os
import pandas as pd
from sqlalchemy import create_engine

db_host = os.getenv("POSTGRES_HOST")
db_user = os.getenv("POSTGRES_USER")
db_password = os.getenv("POSTGRES_PASSWORD")
db_name = os.getenv("POSTGRES_DB")

db_uri = f'postgresql+psycopg2://{db_user}:{db_password}@{db_host}:5432/{db_name}'
engine = create_engine(db_uri)

def export_table_to_csv(table_name: str):
    try:
        query = f'SELECT * FROM {table_name};'
        df = pd.read_sql(query, engine)

        csv_file_path = f'{table_name}.csv'
        df.to_csv(csv_file_path, index=False)
        print(f"Data exported to {csv_file_path}")
    except Exception as e:
        print(f"Failed to export data: {e}")

export_table_to_csv('telegram_ru')