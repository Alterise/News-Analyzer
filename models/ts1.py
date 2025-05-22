import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from tqdm import tqdm
from embedding_storage import EmbeddingStorage
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
db_config = {
    "dbname": os.getenv("POSTGRES_DB"),
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD"),
    "host": os.getenv("POSTGRES_HOST"),
    "port": os.getenv("POSTGRES_PORT", "5432"),
}

class FinancialDataset(Dataset):
    def __init__(self, X_ts, y, X_news=None):
        self.X_ts = torch.FloatTensor(X_ts)
        self.y = torch.FloatTensor(y)
        self.X_news = torch.FloatTensor(X_news) if X_news is not None else None
        self.has_news = X_news is not None
        
    def __len__(self):
        return len(self.X_ts)
    
    def __getitem__(self, idx):
        if self.has_news:
            return (self.X_ts[idx], self.X_news[idx]), self.y[idx]
        return self.X_ts[idx], self.y[idx]

class DataPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def convert_volume(self, val):
        """Convert string with K/M/B suffix to float"""
        if pd.isna(val):
            return 0.0
        val = str(val).strip().upper()
        multipliers = {'K': 1e3, 'M': 1e6, 'B': 1e9}
        for suffix, mult in multipliers.items():
            if suffix in val:
                return float(val.replace(suffix, '')) * mult
        return float(val)

    def clean_numeric(self, val):
        """Clean any numeric string with commas/percentages"""
        if pd.isna(val):
            return 0.0
        return float(str(val).replace(',', '').replace('%', ''))

    def load_data(self, file_path):
        """Load and preprocess financial data"""
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Clean and convert numeric columns
        df['Vol.'] = df['Vol.'].apply(self.convert_volume)
        numeric_cols = ['Price', 'Open', 'High', 'Low', 'Vol.', 'Change %']
        
        for col in numeric_cols:
            if col != 'Vol.':
                df[col] = df[col].apply(self.clean_numeric)
        
        # Normalization
        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        return df.sort_values('Date')

    def get_news_embeddings(self, start_date, end_date, model_name='ai-forever/sbert_large_nlu_ru'):
        """Retrieve news embeddings from database"""
        with EmbeddingStorage(db_config) as storage:
            embeddings = storage.get_all_embeddings_for_model(model_name)
        
        # Convert to DataFrame with date and embedding
        news_data = []
        for timestamp, embedding in embeddings:
            if start_date <= timestamp.date() <= end_date:
                news_data.append({
                    'date': timestamp.date(),
                    'embedding': embedding
                })
        
        if not news_data:
            return None
            
        return pd.DataFrame(news_data)

    def create_sequences(self, data, lookback, news_data=None):
        """Create time series sequences with news embeddings"""
        X_ts, X_news, y = [], [], []
        
        for i in range(lookback, len(data)):
            current_date = data.iloc[i]['Date'].date()
            X_ts.append(data.iloc[i-lookback:i][['Price', 'Open', 'High', 'Low', 'Vol.', 'Change %']].values)
            
            # Add news embeddings if available
            if news_data is not None:
                news_row = news_data[news_data['date'] == current_date]
                if not news_row.empty:
                    X_news.append(news_row['embedding'].mean())
                else:
                    # Zero-padding if no news for this date
                    X_news.append(np.zeros(1024))  # sbert_large_nlu_ru embedding size
            
            y.append(data.iloc[i]['Price'])
        
        X_ts = np.array(X_ts)
        y = np.array(y)
        
        if news_data is not None:
            X_news = np.array(X_news)
            return X_ts, X_news, y
        return X_ts, y

class FairTimeSeriesModel(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.regressor(out[:, -1, :])

class FairCombinedModel(nn.Module):
    def __init__(self, ts_input_size, embedding_size, hidden_size=64):
        super().__init__()
        # Identical time series processing
        self.ts_lstm = nn.LSTM(
            input_size=ts_input_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        
        # Minimal news processing
        self.news_proj = nn.Linear(embedding_size, hidden_size)
        
        # Combined processing (only difference)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),  # Only changed part
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, ts_x, news_x):
        # Identical time series processing
        ts_out, _ = self.ts_lstm(ts_x)
        ts_context = ts_out[:, -1, :]
        
        # News processing
        news_context = self.news_proj(news_x)
        
        # Concatenate and predict
        combined = torch.cat([ts_context, news_context], dim=1)
        return self.regressor(combined)

def evaluate(model, dataloader, criterion, device, is_combined=False):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            if is_combined:
                (X_ts, X_news), y = batch
                X_ts, X_news = X_ts.to(device), X_news.to(device)
                outputs = model(X_ts, X_news)
            else:
                X_ts, y = batch
                X_ts = X_ts.to(device)
                outputs = model(X_ts)
            
            y = y.to(device)
            loss = criterion(outputs, y.unsqueeze(1))
            total_loss += loss.item()
            
            all_preds.extend(outputs.squeeze().cpu().numpy())
            all_targets.extend(y.cpu().numpy())
    
    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()
    
    metrics = {
        'loss': total_loss / len(dataloader),
        'mae': mean_absolute_error(all_targets, all_preds),
        'rmse': np.sqrt(mean_squared_error(all_targets, all_preds))
    }
    return metrics, all_preds, all_targets

def train_model(model, train_loader, val_loader, device, is_combined=False, epochs=100):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_loss = float('inf')
    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            if is_combined:
                (X_ts, X_news), y = batch
                X_ts, X_news = X_ts.to(device), X_news.to(device)
                optimizer.zero_grad()
                outputs = model(X_ts, X_news)
            else:
                X_ts, y = batch
                X_ts = X_ts.to(device)
                optimizer.zero_grad()
                outputs = model(X_ts)
            
            y = y.to(device)
            loss = criterion(outputs, y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Correctly unpack the evaluation results
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device, is_combined)
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save(model.state_dict(), 'best_model.pth' if not is_combined else 'best_combined_model.pth')
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth' if not is_combined else 'best_combined_model.pth'))
    return model

def plot_comparison(ts_true, ts_preds, cmb_true, cmb_preds, scaler, save_path="comparison.png"):
    """Plot both models' predictions together"""
    # Inverse transform using dummy array
    def inverse_transform(values):
        dummy = np.zeros((len(values), 6))
        dummy[:, 0] = values
        return scaler.inverse_transform(dummy)[:, 0]
    
    ts_true_orig = inverse_transform(ts_true)
    ts_preds_orig = inverse_transform(ts_preds)
    cmb_preds_orig = inverse_transform(cmb_preds)
    
    plt.figure(figsize=(14, 7))
    plt.plot(ts_true_orig, label='Actual Price', color='black', linestyle='--', alpha=0.7)
    plt.plot(ts_preds_orig, label='Time Series Model', color='blue', alpha=0.8)
    plt.plot(cmb_preds_orig, label='Combined Model', color='red', alpha=0.8)
    
    plt.title('Model Performance Comparison')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        try:
            plt.show()
        except:
            plt.savefig("comparison_fallback.png", dpi=300, bbox_inches='tight')
            plt.close()

def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lookback = 30
    batch_size = 64
    epochs = 100
    test_size = 0.2
    
    # Load and preprocess data
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data('btc_usd.csv')
    
    # Get news embeddings
    start_date = df['Date'].min().date()
    end_date = df['Date'].max().date()
    news_data = preprocessor.get_news_embeddings(start_date, end_date)
    
    # Create sequences
    if news_data is not None:
        X_ts, X_news, y = preprocessor.create_sequences(df, lookback, news_data)
        embedding_size = 1024  # sbert_large_nlu_ru embedding size
    else:
        print("No news embeddings found")
        X_ts, y = preprocessor.create_sequences(df, lookback)
        X_news = None
    
    # Split data (maintain temporal order)
    indices = np.arange(len(X_ts))
    X_train, X_test, y_train, y_test = train_test_split(
        indices, y, test_size=test_size, shuffle=False
    )
    
    # Create datasets
    ts_train_dataset = FinancialDataset(X_ts[X_train], y_train)
    ts_test_dataset = FinancialDataset(X_ts[X_test], y_test)
    
    if news_data is not None:
        cmb_train_dataset = FinancialDataset(X_ts[X_train], y_train, X_news[X_train])
        cmb_test_dataset = FinancialDataset(X_ts[X_test], y_test, X_news[X_test])
    
    # Create dataloaders
    ts_train_loader = DataLoader(ts_train_dataset, batch_size=batch_size, shuffle=True)
    ts_test_loader = DataLoader(ts_test_dataset, batch_size=batch_size)
    
    if news_data is not None:
        cmb_train_loader = DataLoader(cmb_train_dataset, batch_size=batch_size, shuffle=True)
        cmb_test_loader = DataLoader(cmb_test_dataset, batch_size=batch_size)
    
    # Initialize and train models
    print("Training Time Series Model...")
    ts_model = FairTimeSeriesModel(input_size=6).to(device)
    ts_model = train_model(ts_model, ts_train_loader, ts_test_loader, device, epochs=epochs)
    
    if news_data is not None:
        print("\nTraining Combined Model...")
        cmb_model = FairCombinedModel(ts_input_size=6, embedding_size=1024).to(device)
        cmb_model = train_model(cmb_model, cmb_train_loader, cmb_test_loader, device, is_combined=True, epochs=epochs)
    
    print("\nEpochs:", epochs)

    # Evaluate
    ts_metrics, ts_preds, ts_true = evaluate(ts_model, ts_test_loader, nn.MSELoss(), device)
    print("\nTime Series Model Results:")
    print(f"Test Loss: {ts_metrics['loss']:.4f}")
    print(f"MAE: {ts_metrics['mae']:.4f}")
    print(f"RMSE: {ts_metrics['rmse']:.4f}")
    
    if news_data is not None:
        cmb_metrics, cmb_preds, cmb_true = evaluate(cmb_model, cmb_test_loader, nn.MSELoss(), device, is_combined=True)
        print("\nCombined Model Results:")
        print(f"Test Loss: {cmb_metrics['loss']:.4f}")
        print(f"MAE: {cmb_metrics['mae']:.4f}")
        print(f"RMSE: {cmb_metrics['rmse']:.4f}")
        
        # Calculate improvements
        improvement = {
            'loss': (ts_metrics['loss'] - cmb_metrics['loss']) / ts_metrics['loss'] * 100,
            'mae': (ts_metrics['mae'] - cmb_metrics['mae']) / ts_metrics['mae'] * 100,
            'rmse': (ts_metrics['rmse'] - cmb_metrics['rmse']) / ts_metrics['rmse'] * 100
        }
        print("\nImprovement Comparison:")
        print(f"Loss Improvement: {improvement['loss']:.2f}%")
        print(f"MAE Improvement: {improvement['mae']:.2f}%")
        print(f"RMSE Improvement: {improvement['rmse']:.2f}%")
        
        # Plot comparison
        plot_comparison(ts_true, ts_preds, cmb_true, cmb_preds, preprocessor.scaler)

if __name__ == '__main__':
    main()