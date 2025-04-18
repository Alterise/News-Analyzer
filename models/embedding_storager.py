import pandas as pd
import torch
from tqdm import tqdm
from datetime import datetime
from embedding_storage import EmbeddingStorage
from embedding_model_bert import EmbeddingModel
import os
from dotenv import load_dotenv
import numpy as np

# Конфигурация
load_dotenv()
db_config = {
    "dbname": os.getenv("POSTGRES_DB"),
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD"),
    "host": os.getenv("POSTGRES_HOST"),
    "port": os.getenv("POSTGRES_PORT", "5432"),
}

def clean_text(text):
    return str(text).strip() if pd.notna(text) else ""

def process_data(model_name, use_sbert=False, batch_size=16):
    # Load and preprocess data
    print("Loading and preprocessing data...")
    news_df = pd.read_csv('telegram_en_fin.csv')
    news_df['date'] = pd.to_datetime(news_df['date'])
    news_df['special_id'] = news_df['channel_id'].astype(str) + '_' + news_df['message_id'].astype(str)
    news_df['text'] = news_df['text'].apply(clean_text)
    news_df = news_df[news_df['text'].str.len() > 0].copy()  # Ensure we're working with a copy

    # Initialize embedding model
    print("Loading embedding model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embedding_model = EmbeddingModel(model_name, device, use_sbert=use_sbert)

    # Process embeddings with alignment check
    embeddings = []
    valid_indices = []
    
    for i in tqdm(range(0, len(news_df), batch_size), desc="Batch processing"):
        batch = news_df.iloc[i:i+batch_size]
        texts = batch['text'].tolist()
        
        try:
            batch_embeddings = embedding_model.encode_text(texts)
            if len(batch_embeddings) != len(texts):
                print(f"Warning: Batch {i//batch_size} returned {len(batch_embeddings)} embeddings for {len(texts)} texts")
                continue
                
            embeddings.extend(batch_embeddings)
            valid_indices.extend(batch.index.tolist())
            
        except Exception as e:
            print(f"Error in batch {i//batch_size}: {str(e)}")
            continue

    # Create aligned DataFrame
    processed_df = news_df.loc[valid_indices].copy()
    processed_df['embeddings'] = embeddings
    processed_df['embedding_size'] = processed_df['embeddings'].apply(len)

    # Final verification
    if len(processed_df) != len(embeddings):
        print(f"Critical error: Alignment failed! DataFrame: {len(processed_df)}, Embeddings: {len(embeddings)}")
        # Fallback - take min length
        min_len = min(len(processed_df), len(embeddings))
        processed_df = processed_df.iloc[:min_len].copy()
        processed_df['embeddings'] = embeddings[:min_len]
        processed_df['embedding_size'] = processed_df['embeddings'].apply(len)

    # Then save to database
    embeddings_batch = [
        (row['special_id'], model_name, row['text'],
        row['embeddings'].tolist(), row['embedding_size'], row['date'])
        for _, row in processed_df.iterrows()
    ]

    with EmbeddingStorage(db_config) as storage:
        storage.save_embeddings_batch(embeddings_batch)

    print(f"Processed {len(processed_df)}/{len(news_df)} rows successfully")

# process_data('ai-forever/sbert_large_nlu_ru', use_sbert=True, batch_size=192)
process_data('ProsusAI/finbert', use_sbert=False)