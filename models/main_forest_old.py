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

random_state = 42

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

def aggregate_embeddings_mean(embeddings_df):
    embeddings_df['date'] = pd.to_datetime(embeddings_df['timestamp']).dt.date
    
    grouped = embeddings_df.groupby('date')['embedding_vector'].apply(
        lambda x: np.mean(np.vstack(x.values), axis=0)
    ).reset_index()
    
    grouped.columns = ['timestamp', 'embedding_vector']
    
    return grouped

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
    
    embeddings_agg = aggregate_embeddings_mean(embeddings_df)
    
    currency_df = load_currency_data('usd_rub.csv')
    
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
    plt.savefig("mean_aggregated_tsne.png")
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
    plt.savefig("mean_predictions_tsne.png")
    plt.close()