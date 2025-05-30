import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import psycopg2
import os
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.manifold import TSNE
from dotenv import load_dotenv
from psycopg2.extras import execute_values
import optuna
from optuna.samplers import TPESampler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

random_state = 42

load_dotenv()

db_config = {
    "dbname": os.getenv("POSTGRES_DB"),
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD"),
    "host": os.getenv("POSTGRES_HOST"),
    "port": os.getenv("POSTGRES_PORT", "5432"),
}

class NewsAggregator(nn.Module):
    def __init__(self, input_size, hidden_size=1024, num_layers=2, dropout=0.2):
        super(NewsAggregator, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x, lengths):
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        packed_output, hidden = self.gru(packed_x)
        
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        batch_size, max_len, hidden_size = output.shape
        mask = torch.arange(max_len).expand(batch_size, max_len).to(output.device) < lengths.unsqueeze(1)
        
        attention_scores = self.attention(output).squeeze(-1)
        
        attention_scores = attention_scores.masked_fill(~mask, -1e10)
        
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        weighted_output = torch.bmm(attention_weights.unsqueeze(1), output).squeeze(1)
        
        return weighted_output, attention_weights

def get_embeddings_from_db(model_name):
    query = """
        SELECT timestamp, embedding_vector, special_id, input_text
        FROM embeddings
        WHERE model_name = %s;
    """
    try:
        with psycopg2.connect(**db_config) as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (model_name,))
                return cursor.fetchall()
    except Exception as e:
        print(f"Error retrieving embeddings: {e}")
        return []

def convert_to_numeric_array(x):
    if isinstance(x, str):
        return np.array([float(i) for i in x.strip('[]').split(',')])
    return x

def aggregate_embeddings_rnn(embeddings_df, device='cuda' if torch.cuda.is_available() else 'cpu'):
    embeddings_df['date'] = pd.to_datetime(embeddings_df['timestamp']).dt.date
    embeddings_df = embeddings_df.sort_values(['date', 'timestamp'])
    
    embedding_dim = len(embeddings_df['embedding_vector'].iloc[0])
    
    grouped = embeddings_df.groupby('date')
    
    model = NewsAggregator(input_size=embedding_dim).to(device)
    model.eval()
    
    date_embeddings = []
    
    for date, group in tqdm(grouped, desc="Aggregating with RNN"):
        if len(group) == 1:
            date_embeddings.append((date, group['embedding_vector'].iloc[0]))
            continue
            
        day_embeddings = np.vstack(group['embedding_vector'].values)
        
        embeddings_tensor = torch.tensor(day_embeddings, dtype=torch.float32).unsqueeze(0).to(device)
        seq_length = torch.tensor([embeddings_tensor.shape[1]], dtype=torch.long).to(device)
        
        with torch.no_grad():
            agg_embedding, _ = model(embeddings_tensor, seq_length)
            
        date_embeddings.append((date, agg_embedding.squeeze(0).cpu().numpy()))
    
    result_df = pd.DataFrame(date_embeddings, columns=['timestamp', 'embedding_vector'])
    return result_df

def load_currency_data(filepath):
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y').dt.date
    df['Next Day Change %'] = df['Change %'].str.rstrip('%').astype(float).shift(-1)
    return df.dropna(subset=['Next Day Change %'])

def calculate_feature_importance(model, X, feature_names=None):
    if not feature_names:
        feature_names = [f'Feature {i}' for i in range(X.shape[1])]
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importances')
    plt.bar(range(20), importances[indices[:20]], align='center')
    plt.xticks(range(20), [feature_names[i] for i in indices[:20]], rotation=90)
    plt.tight_layout()
    plt.savefig("feature_importances.png")
    plt.close()
    
    return [(feature_names[i], importances[i]) for i in indices]

import gc
import psutil
import torch

def train_rnn_aggregator(embeddings_df, embedding_dim, device='cuda' if torch.cuda.is_available() else 'cpu'):
    print("Training RNN aggregator with enhanced memory management...")
    
    def print_memory_status():
        if device == 'cuda':
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
        else:
            process = psutil.Process()
            mem = process.memory_info().rss / 1024**3
            print(f"CPU Memory: {mem:.2f}GB used")

    try:
        test_tensor = torch.zeros(1).to(device)
        del test_tensor
    except RuntimeError:
        print("CUDA initialization failed, falling back to CPU")
        device = 'cpu'
    
    try:
        print("Phase 1: Preparing data...")
        embeddings_df['date'] = pd.to_datetime(embeddings_df['timestamp']).dt.date
        
        sequences = []
        lengths = []
        chunk_size = 500
        date_chunks = [embeddings_df['date'].unique()[i:i + chunk_size] 
                      for i in range(0, len(embeddings_df['date'].unique()), chunk_size)]
        
        for i, chunk in enumerate(date_chunks):
            print(f"Processing chunk {i+1}/{len(date_chunks)}")
            chunk_df = embeddings_df[embeddings_df['date'].isin(chunk)]
            
            for date, group in chunk_df.groupby('date'):
                if len(group) > 1:
                    group = group.sort_values('timestamp')
                    day_embeddings = np.vstack(group['embedding_vector'].values)
                    sequences.append(day_embeddings)
                    lengths.append(len(group))
            
            del chunk_df
            gc.collect()
            torch.cuda.empty_cache() if device == 'cuda' else None
            print_memory_status()
        
        if not sequences:
            raise ValueError("No valid sequences found (all days have ≤1 news item)")
        
        max_seq_len = max(lengths)
        print(f"Max sequence length: {max_seq_len}")
        
        print("Phase 2: Converting to tensors...")
        
        tensor_batches = []
        convert_batch_size = 100
        for i in range(0, len(sequences), convert_batch_size):
            batch_seqs = sequences[i:i + convert_batch_size]
            
            batch_array = np.zeros((len(batch_seqs), max_seq_len, embedding_dim), dtype=np.float32)
            for j, seq in enumerate(batch_seqs):
                batch_array[j, :len(seq)] = seq
            
            tensor_batches.append(torch.tensor(batch_array, dtype=torch.float32))
            
            del batch_array, batch_seqs
            gc.collect()
            print_memory_status()
        
        X_seq = torch.cat(tensor_batches)
        X_len = torch.tensor(lengths, dtype=torch.long)
        
        del sequences, tensor_batches
        gc.collect()
        torch.cuda.empty_cache()
        
        print("Phase 3: Setting up training...")
        
        class SeqDataset(torch.utils.data.Dataset):
            def __init__(self, sequences, lengths):
                self.sequences = sequences
                self.lengths = lengths
            
            def __len__(self):
                return len(self.sequences)
            
            def __getitem__(self, idx):
                return self.sequences[idx], self.lengths[idx]
        
        dataset = SeqDataset(X_seq, X_len)
        
        def collate_fn(batch):
            batch.sort(key=lambda x: x[1], reverse=True)
            seqs, lengths = zip(*batch)
            return torch.stack(seqs), torch.stack(lengths)
        
        initial_batch_size = 4
        dataloader = DataLoader(
            dataset,
            batch_size=initial_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=device == 'cuda',
            num_workers=0,
            persistent_workers=False
        )
        
        print("Phase 4: Initializing model...")
        model = NewsAggregator(input_size=embedding_dim).to(device)
        print_memory_status()
        
        print("Phase 5: Starting training...")
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        epochs = 10
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            model.train()
            total_loss = 0
            processed_samples = 0
            
            for batch_idx, (batch_x, batch_len) in enumerate(dataloader):
                try:
                    if batch_idx % 5 == 0:
                        print(f"Batch {batch_idx}, ", end="")
                        print_memory_status()
                    
                    batch_x = batch_x.to(device, non_blocking=True)
                    batch_len = batch_len.to(device, non_blocking=True)
                    
                    aggregated, _ = model(batch_x, batch_len)
                    
                    mask = torch.arange(batch_x.size(1), device=device) < batch_len.unsqueeze(1)
                    mask = mask.unsqueeze(-1).float()
                    target = (batch_x * mask).sum(dim=1) / mask.sum(dim=1)
                    
                    loss = criterion(aggregated, target)
                    
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
                    processed_samples += batch_x.size(0)
                    
                    del batch_x, batch_len, aggregated, target, loss
                    
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        print("\nOOM detected! Handling...")
                        torch.cuda.empty_cache()
                        gc.collect()
                        
                        new_batch_size = max(1, dataloader.batch_size // 2)
                        if new_batch_size != dataloader.batch_size:
                            print(f"Reducing batch size from {dataloader.batch_size} to {new_batch_size}")
                            dataloader = DataLoader(
                                dataset,
                                batch_size=new_batch_size,
                                shuffle=True,
                                collate_fn=collate_fn,
                                pin_memory=device == 'cuda',
                                num_workers=0
                            )
                            break
                        else:
                            raise
                    else:
                        raise
                
                except Exception as e:
                    print(f"Error in batch {batch_idx}: {str(e)}")
                    raise
            
            avg_loss = total_loss / (processed_samples / initial_batch_size) if processed_samples > 0 else float('inf')
            print(f"Epoch {epoch+1} complete. Avg loss: {avg_loss:.4f}")
        
        print("Training completed successfully!")
        return model
    
    except Exception as e:
        print(f"Training failed: {str(e)}")
        torch.cuda.empty_cache()
        gc.collect()
        raise

def objective(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'random_state': random_state
    }
    
    model = RandomForestClassifier(**params)
    return cross_val_score(model, X, y, cv=5, scoring='f1_weighted').mean()

if __name__ == "__main__":
    print("Loading data...")
    model_name = 'ai-forever/sbert_large_nlu_ru'
    embeddings_data = get_embeddings_from_db(model_name)
    if not embeddings_data:
        print("No embeddings found.")
        exit()

    embeddings_df = pd.DataFrame(embeddings_data, columns=['timestamp', 'embedding_vector', 'special_id', 'input_text'])
    embeddings_df['embedding_vector'] = embeddings_df['embedding_vector'].apply(convert_to_numeric_array)
    
    embedding_dim = len(embeddings_df['embedding_vector'].iloc[0])
    
    rnn_model = train_rnn_aggregator(embeddings_df, embedding_dim)
    
    embeddings_agg = aggregate_embeddings_rnn(embeddings_df)
    
    currency_df = load_currency_data('btc_usd.csv')
    
    merged_df = pd.merge(embeddings_agg, currency_df, left_on='timestamp', right_on='Date', how='inner')
    merged_df['target'] = (merged_df['Next Day Change %'] > 0).astype(int)
    
    class_counts = merged_df['target'].value_counts()
    print("Class distribution:")
    print(f"Class 0 (Decrease): {class_counts.get(0, 0)} samples")
    print(f"Class 1 (Increase): {class_counts.get(1, 0)} samples")
    
    X = np.vstack(merged_df['embedding_vector'].values)
    y = merged_df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
    
    print("\nOptimizing hyperparameters with Optuna...")
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=random_state))
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=10)
    
    best_params = study.best_trial.params
    print(f"\nBest parameters: {best_params}")
    best_model = RandomForestClassifier(**best_params, random_state=random_state)
    best_model.fit(X_train, y_train)
    
    joblib.dump(best_model, "best_rf_model.joblib")
    
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    print("\nCalculating feature importance...")
    feature_importances = calculate_feature_importance(best_model, X)
    
    print("\nGenerating t-SNE visualization...")
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=min(30, len(X)-1))
    X_tsne = tsne.fit_transform(X)
    
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.7, edgecolors='w', linewidth=0.5)
    plt.colorbar(scatter, label='1=Increase, 0=Decrease')
    plt.title("t-SNE of News Embeddings\n(Colored by Next Day USD/RUB Change)")
    plt.tight_layout()
    plt.savefig("rnn_aggregated_tsne.png")
    plt.close()
    
    plt.figure(figsize=(12, 10))
    for i, (label, color) in enumerate(zip(['Decrease', 'Increase'], ['red', 'green'])):
        idx = np.where(y == i)[0]
        plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], c=color, label=f'True {label}', alpha=0.5)
    
    test_indices = np.random.choice(range(len(X_test)), size=min(100, len(X_test)), replace=False)
    test_tsne = TSNE(n_components=2, random_state=random_state).fit_transform(X_test[test_indices])

    test_preds = y_pred[test_indices]
        
    for i, (label, marker) in enumerate(zip(['Decrease', 'Increase'], ['x', 'o'])):
        idx = np.where(test_preds == i)[0]
        plt.scatter(test_tsne[idx, 0], test_tsne[idx, 1], 
                    marker=marker, s=100, edgecolors='black', 
                    facecolors='none', linewidth=2, 
                    label=f'Predicted {label}')
    
    plt.legend()
    plt.title("t-SNE of News Embeddings with Predictions")
    plt.tight_layout()
    plt.savefig("rnn_predictions_tsne.png")
    plt.close()