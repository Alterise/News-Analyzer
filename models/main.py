import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import psycopg2
import os
from sklearn.model_selection import cross_val_score, train_test_split, TimeSeriesSplit
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from dotenv import load_dotenv
import optuna
from optuna.samplers import TPESampler

load_dotenv()

db_config = {
    "dbname": os.getenv("POSTGRES_DB"),
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD"),
    "host": os.getenv("POSTGRES_HOST"),
    "port": os.getenv("POSTGRES_PORT", "5432"),
}

def get_embeddings_from_db(model_name):
    query = """
        SELECT timestamp, embedding_vector 
        FROM embeddings_fin 
        WHERE model_name = %s
        ORDER BY timestamp;
    """
    try:
        with psycopg2.connect(**db_config) as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (model_name,))
                return cursor.fetchall()
    except Exception as e:
        print(f"Database error: {e}")
        return []

def process_embeddings(embeddings_data):
    df = pd.DataFrame(embeddings_data, columns=['timestamp', 'embedding'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    
    df['embedding'] = df['embedding'].apply(
        lambda x: np.array([float(i) for i in x.strip('[]').split(',')]))
    
    embedding_matrix = np.vstack(df['embedding'].values)
    
    pca = PCA(n_components=32)
    reduced_embeddings = pca.fit_transform(embedding_matrix)
    for i in range(reduced_embeddings.shape[1]):
        df[f'emb_pca_{i}'] = reduced_embeddings[:, i]
    
    return df

def load_currency_data(filepath):
    df = pd.read_csv(filepath)
    
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y').dt.date
    
    numeric_cols = ['Price', 'Open', 'High', 'Low']
    for col in numeric_cols:
        df[col] = df[col].astype(str).str.replace(',', '').astype(float)
    
    df['Change %'] = df['Change %'].astype(str).str.replace('%', '').replace('', '0').astype(float)
    
    df['Vol.'] = df['Vol.'].replace('', '0').astype(float)
    
    df['price_change'] = df['Change %'].shift(-1)
    df['target'] = (df['price_change'] > 0).astype(int)
    
    return df.dropna(subset=['target'])

def prepare_features(embeddings_df, currency_df):
    merged = pd.merge(embeddings_df, currency_df, left_on='date', right_on='Date')
    
    feature_cols = [col for col in merged.columns if col.startswith('emb_pca_')]
    
    X = merged[feature_cols]
    y = merged['target']
    
    return X, y

def objective(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
    }
    
    model = GradientBoostingClassifier(**params, random_state=42)
    return cross_val_score(model, X, y, cv=TimeSeriesSplit(n_splits=5), scoring='accuracy').mean()

if __name__ == "__main__":
    print("Loading data...")
    embeddings_data = get_embeddings_from_db('ProsusAI/finbert')
    # embeddings_data = get_embeddings_from_db('ai-forever/sbert_large_nlu_ru')
    if not embeddings_data:
        print("No embeddings found in database")
        exit()

    embeddings_df = process_embeddings(embeddings_data)
    currency_df = load_currency_data('usd_rub.csv')
    
    print("Preparing features...")
    X, y = prepare_features(embeddings_df, currency_df)
    
    print("Optimizing model...")
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(lambda trial: objective(trial, X, y), n_trials=30)
    
    print("\nBest hyperparameters:")
    print(study.best_params)
    
    best_model = GradientBoostingClassifier(
        **study.best_params,
        random_state=42
    )
    best_model.fit(X, y)
    
    print("\nModel evaluation:")
    cv_scores = cross_val_score(best_model, X, y, cv=TimeSeriesSplit(n_splits=5))
    print(f"Cross-validation accuracy: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")
    
    plt.figure(figsize=(12, 8))
    importances = best_model.feature_importances_
    features = X.columns
    sorted_idx = np.argsort(importances)[-20:]
    
    plt.barh(range(len(sorted_idx)), importances[sorted_idx])
    plt.yticks(range(len(sorted_idx)), [features[i] for i in sorted_idx])
    plt.title("Top 20 Important Features")
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    print("\nAnalysis complete. Feature importance plot saved.")