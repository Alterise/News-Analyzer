import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import psycopg2
import os
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.manifold import TSNE
from dotenv import load_dotenv
from psycopg2.extras import execute_values
import optuna
from optuna.samplers import TPESampler

# Load environment variables from .env file
load_dotenv()

# Database configuration from environment variables
db_config = {
    "dbname": os.getenv("POSTGRES_DB"),
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD"),
    "host": os.getenv("POSTGRES_HOST"),
    "port": os.getenv("POSTGRES_PORT", "5432"),
}

def get_embeddings_from_db(model_name):
    """Retrieve embeddings from PostgreSQL database."""
    query = """
        SELECT timestamp, embedding_vector
        FROM embeddings_fin
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
    """Convert string array to NumPy array."""
    if isinstance(x, str):
        return np.array([float(i) for i in x.strip('[]').split(',')])
    return x

def aggregate_embeddings(embeddings_data):
    """Aggregate embeddings by date."""
    df = pd.DataFrame(embeddings_data, columns=['timestamp', 'embedding_vector'])
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.date
    df['embedding_vector'] = df['embedding_vector'].apply(convert_to_numeric_array)
    return df.groupby('timestamp')['embedding_vector'].apply(
        lambda x: np.mean(np.vstack(x), axis=0)
    ).reset_index()

def load_currency_data(filepath):
    """Load and preprocess currency data."""
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y').dt.date
    df['Next Day Change %'] = df['Change %'].str.rstrip('%').astype(float).shift(-1)
    return df.dropna(subset=['Next Day Change %'])

def objective(trial, X, y):
    """Optuna objective function for hyperparameter optimization."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'random_state': 42
    }
    
    model = RandomForestClassifier(**params)
    return cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()

if __name__ == "__main__":
    # Data loading and preprocessing
    print("Loading data...")
    # embeddings_data = get_embeddings_from_db('ai-forever/sbert_large_nlu_ru')
    embeddings_data = get_embeddings_from_db('ProsusAI/finbert')
    if not embeddings_data:
        print("No embeddings found.")
        exit()

    embeddings_agg = aggregate_embeddings(embeddings_data)
    currency_df = load_currency_data('usd_rub.csv')
    
    # Merge and prepare data
    merged_df = pd.merge(embeddings_agg, currency_df, left_on='timestamp', right_on='Date', how='inner')
    merged_df['target'] = (merged_df['Next Day Change %'] > 0).astype(int)
    
    X = np.array(merged_df['embedding_vector'].tolist())
    y = merged_df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=38)

    # Optuna optimization
    print("Optimizing hyperparameters with Optuna...")
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=38))
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=50)

    # Train with best params
    print(f"Best trial: {study.best_trial.params}")
    best_model = RandomForestClassifier(**study.best_trial.params, random_state=38)
    best_model.fit(X_train, y_train)

    # Evaluation
    y_pred = best_model.predict(X_test)
    print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Visualization
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.6)
    plt.title("t-SNE of News Embeddings\n(Colored by Next Day USD/RUB Change)")
    plt.colorbar(label='1=Increase, 0=Decrease')
    plt.savefig("optuna_tsne.png")
    plt.close()