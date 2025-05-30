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

load_dotenv()

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
        if pd.isna(val):
            return 0.0
        val = str(val).strip().upper()
        multipliers = {'K': 1e3, 'M': 1e6, 'B': 1e9}
        for suffix, mult in multipliers.items():
            if suffix in val:
                return float(val.replace(suffix, '')) * mult
        return float(val)

    def clean_numeric(self, val):
        if pd.isna(val):
            return 0.0
        return float(str(val).replace(',', '').replace('%', ''))

    def load_data(self, file_path):
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        
        df['Vol.'] = df['Vol.'].apply(self.convert_volume)
        numeric_cols = ['Price', 'Open', 'High', 'Low', 'Vol.', 'Change %']
        
        for col in numeric_cols:
            if col != 'Vol.':
                df[col] = df[col].apply(self.clean_numeric)
        
        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        return df.sort_values('Date')

    def get_news_embeddings(self, start_date, end_date, model_name='ProsusAI/finbert'):
        with EmbeddingStorage(db_config) as storage:
            embeddings = storage.get_all_embeddings_for_model(model_name)
        
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
        X_ts, X_news, y = [], [], []
        
        for i in range(lookback, len(data)):
            current_date = data.iloc[i]['Date'].date()
            X_ts.append(data.iloc[i-lookback:i][['Price', 'Open', 'High', 'Low', 'Vol.', 'Change %']].values)
            
            if news_data is not None:
                news_row = news_data[news_data['date'] == current_date]
                if not news_row.empty:
                    X_news.append(news_row['embedding'].values[0])
                else:
                    if len(news_data) > 0:
                        X_news.append(np.zeros_like(news_data['embedding'].iloc[0]))
                    else:
                        X_news.append(np.zeros(768))
            
            y.append(data.iloc[i]['Price'])
        
        X_ts = np.array(X_ts)
        y = np.array(y)
        
        if news_data is not None:
            X_news = np.array(X_news)
            return X_ts, X_news, y
        return X_ts, y

class TimeSeriesModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        self.linear = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :])

class CombinedModel(nn.Module):
    def __init__(self, ts_input_size, embedding_size, hidden_size=64):
        super().__init__()
        self.ts_lstm = nn.LSTM(
            input_size=ts_input_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.news_encoder = nn.Sequential(
            nn.Linear(embedding_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32)
        )
        self.combined = nn.Sequential(
            nn.Linear(hidden_size + 32, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
        
    def forward(self, ts_x, news_x):
        ts_out, _ = self.ts_lstm(ts_x)
        news_out = self.news_encoder(news_x)
        combined = torch.cat([ts_out[:, -1, :], news_out], dim=1)
        return self.combined(combined)

def train_epoch(model, dataloader, criterion, optimizer, device, is_combined=False):
    model.train()
    total_loss = 0
    for batch in dataloader:
        if is_combined:
            (X_ts, X_news), y = batch
            X_ts, X_news, y = X_ts.to(device), X_news.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X_ts, X_news)
        else:
            X_ts, y = batch
            X_ts, y = X_ts.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X_ts)
        
        loss = criterion(outputs, y.unsqueeze(1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

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

def plot_results(model_name, y_true, y_pred, scaler, feature_index=0, save_path=None):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    dummy_array = np.zeros((len(y_true), 6))
    dummy_array[:, feature_index] = y_true
    y_true_orig = scaler.inverse_transform(dummy_array)[:, feature_index]
    
    dummy_array[:, feature_index] = y_pred
    y_pred_orig = scaler.inverse_transform(dummy_array)[:, feature_index]
    
    plt.figure(figsize=(12, 6))
    plt.plot(y_true_orig, label='Actual Price')
    plt.plot(y_pred_orig, label='Predicted Price')
    plt.title(f'{model_name} - Actual vs Predicted Prices')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        try:
            plt.show()
        except:
            plt.savefig(f"{model_name.replace(' ', '_')}_plot.png")
            plt.close()

def plot_comparison(ts_true, ts_preds, cmb_true, cmb_preds, scaler, feature_index=0, save_path="model_comparison.png"):
    dummy_array = np.zeros((len(ts_true), 6))
    dummy_array[:, feature_index] = ts_true
    ts_true_orig = scaler.inverse_transform(dummy_array)[:, feature_index]
    
    dummy_array[:, feature_index] = ts_preds
    ts_preds_orig = scaler.inverse_transform(dummy_array)[:, feature_index]
    
    dummy_array = np.zeros((len(cmb_true), 6))
    dummy_array[:, feature_index] = cmb_true
    cmb_true_orig = scaler.inverse_transform(dummy_array)[:, feature_index]
    
    dummy_array[:, feature_index] = cmb_preds
    cmb_preds_orig = scaler.inverse_transform(dummy_array)[:, feature_index]
    
    plt.figure(figsize=(14, 7))
    
    plt.plot(ts_true_orig, label='Actual Price', color='black', linestyle='--', alpha=0.7)
    plt.plot(ts_preds_orig, label='Time Series Model', color='blue', alpha=0.8)
    plt.plot(cmb_preds_orig, label='Combined Model', color='red', alpha=0.8)
    
    plt.title('Model Comparison: Actual vs Predicted Prices')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.5)
    
    plt.text(0.02, 0.95, 
             f"Time Series Model\nMAE: {mean_absolute_error(ts_true_orig, ts_preds_orig):.2f}\nRMSE: {np.sqrt(mean_squared_error(ts_true_orig, ts_preds_orig)):.2f}",
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='blue', alpha=0.1))
    
    plt.text(0.02, 0.85, 
             f"Combined Model\nMAE: {mean_absolute_error(cmb_true_orig, cmb_preds_orig):.2f}\nRMSE: {np.sqrt(mean_squared_error(cmb_true_orig, cmb_preds_orig)):.2f}",
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='red', alpha=0.1))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        try:
            plt.show()
        except:
            plt.savefig("model_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lookback = 30
    batch_size = 64
    epochs = 150
    test_size = 0.2
    embedding_model_name = 'ai-forever/sbert_large_nlu_ru'
    
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data('btc_usd.csv')
    
    start_date = df['Date'].min().date()
    end_date = df['Date'].max().date()
    news_data = preprocessor.get_news_embeddings(start_date, end_date, embedding_model_name)
    
    if news_data is not None:
        X_ts, X_news, y = preprocessor.create_sequences(df, lookback, news_data)
        embedding_size = len(news_data['embedding'].iloc[0])
    else:
        print("No news embeddings found in database")
        X_ts, y = preprocessor.create_sequences(df, lookback)
        X_news = None
        embedding_size = 1024
    
    indices = np.arange(len(X_ts))
    X_train, X_test, y_train, y_test = train_test_split(
        indices, y, test_size=test_size, shuffle=False
    )
    
    if news_data is not None:
        train_dataset = FinancialDataset(X_ts[X_train], y_train, X_news[X_train])
        test_dataset = FinancialDataset(X_ts[X_test], y_test, X_news[X_test])
    else:
        train_dataset = FinancialDataset(X_ts[X_train], y_train)
        test_dataset = FinancialDataset(X_ts[X_test], y_test)
    
    ts_train_dataset = FinancialDataset(X_ts[X_train], y_train)
    ts_test_dataset = FinancialDataset(X_ts[X_test], y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    ts_train_loader = DataLoader(ts_train_dataset, batch_size=batch_size, shuffle=True)
    ts_test_loader = DataLoader(ts_test_dataset, batch_size=batch_size)
    
    ts_model = TimeSeriesModel(input_size=6).to(device)
    combined_model = CombinedModel(ts_input_size=6, embedding_size=embedding_size).to(device)
    
    criterion = nn.MSELoss()
    ts_optimizer = optim.Adam(ts_model.parameters(), lr=0.001)
    combined_optimizer = optim.Adam(combined_model.parameters(), lr=0.001)
    
    print("Training Time Series Model...")
    ts_train_losses = []
    ts_val_losses = []
    
    for epoch in tqdm(range(epochs)):
        train_loss = train_epoch(ts_model, ts_train_loader, criterion, ts_optimizer, device)
        val_metrics, _, _ = evaluate(ts_model, ts_test_loader, criterion, device)
        
        ts_train_losses.append(train_loss)
        ts_val_losses.append(val_metrics['loss'])
    
    if news_data is not None:
        print("\nTraining Combined Model...")
        cmb_train_losses = []
        cmb_val_losses = []
        
        for epoch in tqdm(range(epochs)):
            train_loss = train_epoch(combined_model, train_loader, criterion, combined_optimizer, device, is_combined=True)
            val_metrics, _, _ = evaluate(combined_model, test_loader, criterion, device, is_combined=True)
            
            cmb_train_losses.append(train_loss)
            cmb_val_losses.append(val_metrics['loss'])
    
    ts_metrics, ts_preds, ts_true = evaluate(ts_model, ts_test_loader, criterion, device)
    
    if news_data is not None:
        cmb_metrics, cmb_preds, cmb_true = evaluate(combined_model, test_loader, criterion, device, is_combined=True)
    
    print("\nEpochs:", epochs)

    print("\nTime Series Model Results:")
    print(f"Test Loss: {ts_metrics['loss']:.4f}")
    print(f"MAE: {ts_metrics['mae']:.4f}")
    print(f"RMSE: {ts_metrics['rmse']:.4f}")
    
    if news_data is not None:
        print("\nCombined Model Results:")
        print(f"Test Loss: {cmb_metrics['loss']:.4f}")
        print(f"MAE: {cmb_metrics['mae']:.4f}")
        print(f"RMSE: {cmb_metrics['rmse']:.4f}")
        
        print("\nImprovement Comparison:")
        improvement = {
            'loss': (ts_metrics['loss'] - cmb_metrics['loss']) / ts_metrics['loss'] * 100,
            'mae': (ts_metrics['mae'] - cmb_metrics['mae']) / ts_metrics['mae'] * 100,
            'rmse': (ts_metrics['rmse'] - cmb_metrics['rmse']) / ts_metrics['rmse'] * 100
        }
        print(f"Loss Improvement: {improvement['loss']:.2f}%")
        print(f"MAE Improvement: {improvement['mae']:.2f}%")
        print(f"RMSE Improvement: {improvement['rmse']:.2f}%")
    
    plot_results("Time Series Model", ts_true, ts_preds, preprocessor.scaler, 
                feature_index=0, save_path="ts_model_plot.png")
    if news_data is not None:
        plot_comparison(ts_true, ts_preds, cmb_true, cmb_preds, 
                    preprocessor.scaler, feature_index=0,
                    save_path="model_comparison.png")

if __name__ == '__main__':
    main()