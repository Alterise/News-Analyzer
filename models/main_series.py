import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_recall_curve, roc_auc_score
from sklearn.utils import compute_class_weight
from xgboost import XGBClassifier, XGBRegressor, plot_importance
from dotenv import load_dotenv
from embedding_storage import EmbeddingStorage
import re

load_dotenv()

db_config = {
    "dbname": os.getenv("POSTGRES_DB"),
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD"),
    "host": os.getenv("POSTGRES_HOST"),
    "port": os.getenv("POSTGRES_PORT", "5432"),
}

def compute_atr(df, window=14):
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Price']).abs()
    low_close = (df['Low'] - df['Price']).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window).mean()

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def load_and_validate_data(embedding_model, currency_file):
    print("Loading embeddings...")
    with EmbeddingStorage(db_config) as storage:
        embeddings = storage.get_all_embeddings_for_model(embedding_model)
    
    if not embeddings:
        raise ValueError("No embeddings found")
    
    embeddings_df = pd.DataFrame(embeddings, columns=['timestamp', 'embedding'])
    embeddings_df['timestamp'] = pd.to_datetime(embeddings_df['timestamp']).dt.normalize()
    
    print("Loading currency data...")
    try:
        currency_df = pd.read_csv(currency_file)
    except Exception as e:
        raise ValueError(f"Error loading currency data: {str(e)}")
    
    currency_df['Date'] = pd.to_datetime(currency_df['Date']).dt.normalize()
    
    required_cols = {'Date', 'Price', 'Open', 'High', 'Low', 'Change %'}
    if not required_cols.issubset(currency_df.columns):
        missing = required_cols - set(currency_df.columns)
        raise ValueError(f"Missing required columns: {missing}")
    
    for col in ['Price', 'Open', 'High', 'Low']:
        currency_df[col] = currency_df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
        currency_df[col] = pd.to_numeric(currency_df[col], errors='coerce')
    
    currency_df['Change %'] = currency_df['Change %'].astype(str).str.replace(r'[^\d.-]', '', regex=True)
    currency_df['Change %'] = pd.to_numeric(currency_df['Change %'], errors='coerce')
    
    merged_df = pd.merge(
        embeddings_df,
        currency_df,
        left_on='timestamp',
        right_on='Date',
        how='inner'
    )
    
    if len(merged_df) == 0:
        raise ValueError("Merge failed - no overlapping dates")
    
    print(f"\nData validation:")
    print(f"- Total records: {len(merged_df)}")
    print(f"- Date range: {merged_df['Date'].min()} to {merged_df['Date'].max()}")
    print(f"- NA counts:\n{merged_df.isna().sum()}")
    
    return merged_df

def convert_volume(vol_str):
    if pd.isna(vol_str) or vol_str == '-':
        return 0
    try:
        vol_str = str(vol_str).upper().replace(',', '')
        if 'K' in vol_str:
            return float(vol_str.replace('K', '')) * 1e3
        elif 'M' in vol_str:
            return float(vol_str.replace('M', '')) * 1e6
        return float(vol_str)
    except:
        return 0

def create_features(df):
    if 'Vol.' in df.columns:
        df['Vol.'] = (
            df['Vol.'].astype(str)
            .str.replace(r'[^\d.KM]', '', regex=True)
            .replace({'K': '*1e3', 'M': '*1e6'}, regex=True)
            .map(pd.eval)
            .fillna(0)
        )
    
    df['price_ma_ratio'] = df['Price'] / df['Price'].rolling(20).mean().replace(0, np.nan)
    df['rsi_14'] = compute_rsi(df['Price'], 14)
    df['atr_14'] = compute_atr(df, 14)
    
    features = {
        'price_range': ((df['High'] - df['Low']) / df['Price'].replace(0, np.nan)).clip(-10, 10),
        'day_of_week': df['Date'].dt.dayofweek,
        'month': df['Date'].dt.month,
        'volatility_3day': df['Change %'].abs().rolling(3).std().clip(0, 100),
        'volume_change': df['Vol.'].pct_change().fillna(0),
        'momentum_5': df['Price'].pct_change(5).clip(-5, 5)
    }
    
    for name, values in features.items():
        df[name] = values.replace([np.inf, -np.inf], np.nan).fillna(0)
    
        conditions = [
        (df['Change %'].shift(-1) < -1.5),
        (df['Change %'].shift(-1) > 1.5)
    ]
    choices = [0, 2]
    df['target'] = np.select(conditions, choices, default=1)
    
    return df.dropna()

def train_and_evaluate(df):
    feature_cols = [col for col in [
        'price_range', 'day_of_week', 'month',
        'volatility_3day', 'volume_change', 'momentum_5',
        'price_ma_ratio', 'rsi_14', 'atr_14'
    ] if col in df.columns]
    
    X = df[feature_cols]
    y = df['target']
    
    if X.isnull().any().any():
        print(f"Filling {X.isnull().sum().sum()} NA values")
        X = X.fillna(0)
    
    if np.isinf(X.to_numpy()).any():
        raise ValueError("Infinite values in features")
    
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    if isinstance(y.iloc[0], str):
        class_mapping = {'down': 0, 'neutral': 1, 'up': 2}
        y = y.map(class_mapping)
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', XGBClassifier(
            max_depth=4,
            learning_rate=0.05,
            n_estimators=300,
            objective='multi:softmax',
            num_class=3,
            eval_metric='mlogloss',
            random_state=42
        ))
    ])
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    class_names = ['down', 'neutral', 'up']
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    return model


if __name__ == "__main__":
    try:
        print("Starting pipeline...")
        
        df = load_and_validate_data('ProsusAI/finbert', 'btc_usd.csv')
        
        df = create_features(df)
        print("\nSample features:")
        print(df[['Date', 'Price', 'price_range', 'volatility_3day', 'target']].head())
        
        model = train_and_evaluate(df)
        
        df.to_csv('processed_trading_data.csv', index=False)
        print("\nPipeline completed successfully")
        
    except Exception as e:
        print(f"\nPipeline failed: {str(e)}")
        if 'df' in locals():
            print("\nDebug Info:")
            print("Columns:", df.columns.tolist())
            print("Date Range:", df['Date'].min(), "to", df['Date'].max())
            print("NA Counts:\n", df.isna().sum())
            print("Sample Data:\n", df.head())