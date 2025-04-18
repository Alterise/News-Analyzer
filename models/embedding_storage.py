import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
import numpy as np
from datetime import datetime

class EmbeddingStorage:
    def __init__(self, db_config):
        self.db_config = db_config
        self.conn = None
        self.vector_registered = False

    def __enter__(self):
        self.conn = psycopg2.connect(**self.db_config)
        self._register_vector_type()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()

    def _register_vector_type(self):
        """Register the vector type adapter if not already registered"""
        if not self.vector_registered and self.conn:
            register_vector(self.conn)
            self.vector_registered = True

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None
            self.vector_registered = False

    def save_embedding(self, special_id, model_name, input_text, embedding, embedding_size, timestamp=None):
        """
        Save a single embedding to the database.
        
        Args:
            embedding: Should be a numpy array or list that can be converted to pgvector
        """
        if self.conn is None:
            raise ValueError("Database connection is not established.")

        try:
            with self.conn.cursor() as cursor:
                if timestamp is None:
                    timestamp = datetime.now()

                # Convert to numpy array if not already
                if not isinstance(embedding, np.ndarray):
                    embedding = np.array(embedding, dtype=np.float32)

                cursor.execute("""
                    INSERT INTO embeddings 
                    (special_id, model_name, input_text, embedding_vector, embedding_size, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (model_name, special_id) DO NOTHING;
                """, (special_id, model_name, input_text, embedding, embedding_size, timestamp))

                self.conn.commit()
                if cursor.rowcount > 0:
                    print("Embedding saved successfully.")
                else:
                    print(f"Skipped duplicate (model_name: {model_name}, special_id: {special_id})")

        except Exception as e:
            print(f"Error saving embedding to database: {e}")
            self.conn.rollback()

    def save_embeddings_batch(self, embeddings_batch):
        if self.conn is None:
            raise ValueError("Database connection not established")

        try:
            with self.conn.cursor() as cursor:
                # Prepare all data for insertion
                data_to_insert = []
                
                for record in embeddings_batch:
                    # Ensure embedding is numpy array
                    embedding = record[3]
                    if not isinstance(embedding, np.ndarray):
                        embedding = np.array(embedding, dtype=np.float32)

                    data_to_insert.append((
                        str(record[0]),  # special_id
                        str(record[1]),  # model_name
                        str(record[2]),  # input_text
                        embedding,      # embedding_vector
                        int(record[4]), # embedding_size
                        record[5] if record[5] else datetime.now()  # timestamp
                    ))

                # Insert all records in one operation
                if data_to_insert:
                    execute_values(
                        cursor,
                        """INSERT INTO embeddings_fin
                        (special_id, model_name, input_text, embedding_vector, embedding_size, timestamp)
                        VALUES %s
                        ON CONFLICT (special_id, model_name) DO NOTHING""",
                        data_to_insert
                    )
                    self.conn.commit()
                    
                    # Get the count of actually inserted rows
                    inserted_count = cursor.rowcount
                    duplicate_count = len(data_to_insert) - inserted_count
                    
                    print(f"Saved {inserted_count} embeddings")
                    print(f"Skipped {duplicate_count} duplicates")

        except Exception as e:
            self.conn.rollback()
            raise RuntimeError(f"Batch save failed: {str(e)}")
        

    def get_all_embeddings_for_model(self, model_name):
        """
        Retrieve all embeddings for a specified model as numpy arrays.
        
        Returns:
            List of (timestamp, numpy_array) tuples
        """
        if self.conn is None:
            raise ValueError("Database connection is not established.")

        try:
            self._register_vector_type()  # Ensure vector type is registered
            
            with self.conn.cursor() as cursor:
                cursor.execute("""
                    SELECT timestamp, embedding_vector
                    FROM embeddings
                    WHERE model_name = %s;
                """, (model_name,))
                results = cursor.fetchall()

                if results:
                    # Verify we got numpy arrays
                    if not isinstance(results[0][1], np.ndarray):
                        raise RuntimeError("Vector type registration failed - got unexpected type")
                    return results
                else:
                    print(f"No embeddings found for model '{model_name}'.")
                    return []

        except Exception as e:
            print(f"Error retrieving embeddings from database: {e}")
            return []