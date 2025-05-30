import torch
import psycopg2
from transformers import MarianMTModel, MarianTokenizer
from typing import List
from tqdm import tqdm
from dotenv import load_dotenv
import os
from psycopg2.extras import execute_values
import gc

load_dotenv()

model_name = "Helsinki-NLP/opus-mt-ru-en"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = MarianTokenizer.from_pretrained(model_name)
try:
    model = MarianMTModel.from_pretrained(model_name).to(device)
    model.eval()
except Exception as e:
    print(f"Model loading failed: {str(e)}")
    exit(1)

db_config = {
    'host': os.getenv("POSTGRES_HOST"),
    'dbname': os.getenv("POSTGRES_DB"),
    'user': os.getenv("POSTGRES_USER"),
    'password': os.getenv("POSTGRES_PASSWORD"),
    'port': 5432
}

def batch_translate(texts: List[str], batch_size: int = 32) -> List[str]:
    translations = []
    
    gen_kwargs = {
        'max_new_tokens': 490,
        'num_beams': 1,
        'do_sample': False,
        'early_stopping': False,
        'pad_token_id': tokenizer.pad_token_id,
        'eos_token_id': tokenizer.eos_token_id
    }
    
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding='longest',
                truncation=True,
                max_length=490,
                return_attention_mask=False
            ).to(device, non_blocking=True)
            
            if inputs['input_ids'].shape[1] == 0:
                translations.extend([""] * len(batch))
                continue
                
            try:
                outputs = model.generate(**inputs, **gen_kwargs)
                translations.extend(tokenizer.batch_decode(
                    outputs,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                ))
            except Exception as e:
                print(f"Error in batch {i//batch_size}: {str(e)}")
                translations.extend([""] * len(batch))
            
            del inputs, outputs
            if device == "cuda":
                torch.cuda.empty_cache()
            if i % (10 * batch_size) == 0:
                gc.collect()
    
    return translations

def translate_and_save_messages(db_batch_size: int = 256, translate_batch: int = 32):
    conn = None
    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT COUNT(*) FROM telegram_ru r
        WHERE NOT EXISTS (
            SELECT 1 FROM telegram_en e 
            WHERE e.channel_id = r.channel_id AND e.message_id = r.message_id
        )
        """)
        total = cursor.fetchone()[0]
        
        if total == 0:
            print("No messages to translate")
            return

        with tqdm(total=total, desc="Translating") as pbar:
            while True:
                try:
                    cursor.execute("""
                    SELECT r.channel_id, r.message_id, r.date, r.text
                    FROM telegram_ru r
                    WHERE NOT EXISTS (
                        SELECT 1 FROM telegram_en e 
                        WHERE e.channel_id = r.channel_id AND e.message_id = r.message_id
                    )
                    ORDER BY r.date DESC
                    LIMIT %s
                    """, (db_batch_size,))
                    
                    messages = cursor.fetchall()
                    if not messages:
                        break

                    texts = [msg[3] for msg in messages]
                    translated = batch_translate(texts, translate_batch)

                    try:
                        execute_values(
                            cursor,
                            """INSERT INTO telegram_en 
                            (channel_id, message_id, date, text)
                            VALUES %s ON CONFLICT DO NOTHING""",
                            [(m[0], m[1], m[2], t) for m, t in zip(messages, translated)],
                            page_size=500
                        )
                        conn.commit()
                    except Exception as e:
                        conn.rollback()
                        print(f"DB insert error: {str(e)}")
                        continue

                    pbar.update(len(messages))
                        
                except Exception as e:
                    print(f"Batch processing error: {str(e)}")
                    continue

    except Exception as e:
        print(f"Fatal error: {str(e)}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    translate_and_save_messages(db_batch_size=256, translate_batch=32)