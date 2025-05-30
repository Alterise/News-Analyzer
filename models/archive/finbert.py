from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import torch
import pandas as pd
import random

def load_finbert(model_name="ProsusAI/finbert"):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)
    return model, tokenizer

def create_sentiment_pipeline(model, tokenizer):
    return pipeline("sentiment-analysis", 
                   model=model, 
                   tokenizer=tokenizer,
                   device=0 if torch.cuda.is_available() else -1)

def analyze_sentiment(nlp, text):
    result = nlp(text)
    return {
        "text": text,
        "label": result[0]['label'],
        "score": result[0]['score']
    }

def batch_analyze_sentiment(nlp, texts):
    results = []
    for text in texts:
        try:
            result = analyze_sentiment(nlp, text)
            results.append(result)
        except Exception as e:
            print(f"Error processing text: {text[:50]}... Error: {str(e)}")
            results.append({
                "text": text,
                "label": "ERROR",
                "score": 0.0
            })
    return pd.DataFrame(results)

def load_random_samples_from_csv(file_path, n_samples=5):
    df = pd.read_csv(file_path)
    if len(df) < n_samples:
        print(f"Warning: File contains only {len(df)} rows, using all available")
        n_samples = len(df)
    random_samples = df.sample(n_samples, random_state=42)
    return random_samples['text'].tolist()

if __name__ == "__main__":
    csv_file = "telegram_raw.csv"
    print(f"Loading 5 random samples from {csv_file}...")
    try:
        test_texts = load_random_samples_from_csv(csv_file, 100)
    except Exception as e:
        print(f"Error loading CSV file: {str(e)}")
        test_texts = [
            "The company reported strong earnings growth this quarter",
            "Market volatility has increased due to geopolitical risks",
            "The merger deal fell through causing shares to plummet",
            "Innovative product launch expected to drive future revenue",
            "Regulatory concerns are weighing on investor sentiment"
        ]
        print("Using default example texts instead")
    
    print("\nLoading FinBERT model...")
    model, tokenizer = load_finbert()
    nlp = create_sentiment_pipeline(model, tokenizer)
    
    print("\nRunning sentiment analysis on sampled texts...")
    results = batch_analyze_sentiment(nlp, test_texts)
    
    print("\nResults:")
    print(results.to_string(index=False))
    
    print("\nSummary Statistics:")
    print(results['label'].value_counts())