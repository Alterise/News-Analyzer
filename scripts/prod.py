import os
import pandas as pd
import numpy as np
import torch
import joblib
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from telethon import TelegramClient
from telethon.tl.functions.channels import JoinChannelRequest
from typing import List, Dict, Optional

load_dotenv()

class TelegramNewsPredictor:
    def __init__(self):
        self.api_id = os.getenv("TELEGRAM_API_ID")
        self.api_hash = os.getenv("TELEGRAM_API_HASH")
        self.channel_username = 'bbbreaking'
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embedding_model_name = 'ai-forever/sbert_large_nlu_ru'
        self.prediction_model_path = "best_rf_model.joblib"
        
        self.client = None
        self.embedding_model = None
        self.prediction_model = None
        
    async def initialize(self):
        print("Initializing Telegram client...")
        self.client = TelegramClient('news_predictor', self.api_id, self.api_hash)
        await self.client.start()
        
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(
            self.embedding_model_name, 
            device=self.device
        )
        
        print("Loading prediction model...")
        try:
            self.prediction_model = joblib.load(self.prediction_model_path)
        except Exception as e:
            print(f"Error loading prediction model: {e}")
            raise
        
        try:
            await self.client(JoinChannelRequest(self.channel_username))
        except Exception as e:
            print(f"Channel join attempt: {e}")

    async def get_yesterday_news(self) -> List[Dict]:
        utc_now = datetime.now(timezone.utc)
        yesterday = utc_now - timedelta(days=1) + timedelta(hours=3)
        start_date = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = yesterday.replace(hour=23, minute=59, second=59, microsecond=999999)
        print(yesterday)
        print(start_date)
        print(end_date)
        
        messages = []
        print(f"Fetching news from {start_date.date()}...")
        
        try:
            async for message in self.client.iter_messages(
                self.channel_username,
                offset_date=end_date
            ):
                if not hasattr(message, 'date'):
                    continue
                    
                message_date = message.date.replace(tzinfo=timezone.utc) if message.date.tzinfo is None else message.date
                
                if message_date < start_date:
                    break
                    
                if message.text:
                    messages.append({
                        'date': message_date,
                        'text': message.text,
                        'message_id': message.id
                    })
                    
        except Exception as e:
            print(f"Error fetching messages: {e}")
            
        return messages

    def process_news(self, messages: List[Dict]) -> pd.DataFrame:
        if not messages:
            return pd.DataFrame()
            
        df = pd.DataFrame(messages)
        df['text'] = df['text'].str.strip()
        df = df[df['text'].str.len() > 0]
        return df

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])
            
        return self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=32
        )

    def aggregate_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        if len(embeddings) == 0:
            return np.zeros(self.embedding_model.get_sentence_embedding_dimension())
            
        return np.mean(embeddings, axis=0)

    async def predict_market_movement(self) -> Optional[Dict]:
        if not all([self.client, self.embedding_model, self.prediction_model]):
            raise RuntimeError("Components not initialized")
            
        messages = await self.get_yesterday_news()
        if not messages:
            print("No news messages found for yesterday")
            return None
            
        news_df = self.process_news(messages)
        if news_df.empty:
            print("No valid news texts after processing")
            return None
            
        embeddings = self.generate_embeddings(news_df['text'].tolist())
        
        daily_embedding = self.aggregate_embeddings(embeddings)
        
        prediction = self.prediction_model.predict([daily_embedding])[0]
        probabilities = self.prediction_model.predict_proba([daily_embedding])[0]
        
        return {
            'date': news_df['date'].iloc[0].date(),
            'news_count': len(news_df),
            'prediction': 'Increase' if prediction == 1 else 'Decrease',
            'confidence': float(probabilities[1]) if prediction == 1 else float(probabilities[0]),
            'sample_news': news_df['text'].head(10).tolist(),
            'embedding_dim': daily_embedding.shape[0]
        }

    async def close(self):
        if self.client:
            await self.client.disconnect()

async def main():
    predictor = TelegramNewsPredictor()
    try:
        await predictor.initialize()
        result = await predictor.predict_market_movement()
        
        if result:
            print("\nPrediction Results:")
            print(f"Date: {result['date']}")
            print(f"News Processed: {result['news_count']}")
            print(f"Predicted Market Movement: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print("\nSample News Headlines:")
            for i, news in enumerate(result['sample_news'], 1):
                print(f"{i}. {news[:100]}{'...' if len(news) > 100 else ''}")
        else:
            print("No prediction results available")
            
    except Exception as e:
        print(f"Pipeline failed: {e}")
    finally:
        await predictor.close()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())