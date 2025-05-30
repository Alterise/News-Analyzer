import io
import os
import psycopg2
import pandas as pd
from tqdm import tqdm
from psycopg2.extras import execute_values
from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.tl.functions.channels import JoinChannelRequest

load_dotenv()
api_id = os.getenv("TELEGRAM_API_ID")
api_hash = os.getenv("TELEGRAM_API_HASH")

db_host = os.getenv("POSTGRES_HOST")
db_user = os.getenv("POSTGRES_USER")
db_password = os.getenv("POSTGRES_PASSWORD")
db_name = os.getenv("POSTGRES_DB")

client = TelegramClient('news_scraping', api_id, api_hash)


async def join_channel(client, channel_link):
    try:
        await client(JoinChannelRequest(channel_link))
        print(f"Successfully joined the channel {channel_link}")
    except Exception as e:
        print(f"Failed to join the channel {channel_link}: {str(e)}")


async def scrape_messages(client, channel, limit=None):
    messages = []
    
    if limit is None:
        try:
            total_messages = await client.get_messages(channel, limit=1)
            total_messages = total_messages.total
            print(f"Found {total_messages} total messages in channel")
        except Exception as e:
            print(f"Couldn't get message count: {str(e)}")
            total_messages = 0
    
    with tqdm(
        total=limit if limit is not None else total_messages,
        desc=f"Scraping {channel.split('/')[-1]}",
        unit="msg",
        dynamic_ncols=True
    ) as pbar:
        try:
            async for message in client.iter_messages(channel, limit=limit):
                if message.text:
                    messages.append({
                        'channel_id': message.peer_id.channel_id,
                        'message_id': message.id,
                        'date': message.date,
                        'text': message.text
                    })
                pbar.update(1)
                
        except Exception as e:
            print(f"\nError during scraping: {str(e)}")
            return messages
    
    print(f'\nSuccessfully scraped {len(messages)} messages from {channel.split("/")[-1]}')
    return messages


def save_to_db(messages, table):
    db = psycopg2.connect(host=db_host, dbname=db_name,
                         user=db_user, password=db_password, port=5432)
    cursor = db.cursor()

    print("Preparing data for database insertion...")
    data = [(msg['channel_id'], msg['message_id'], msg['date'], msg['text']) 
            for msg in tqdm(messages, desc="Preparing records", unit="rec")]

    print("Inserting records to database...")
    execute_values(
        cursor,
        f"""
        INSERT INTO {table} (channel_id, message_id, date, text) 
        VALUES %s
        ON CONFLICT (channel_id, message_id) DO NOTHING;
        """,
        data,
        page_size=1000
    )
    db.commit()
    cursor.close()
    db.close()

    print(f'\nSuccessfully saved {len(messages)} messages to database')


def save_to_csv(messages, channel_name='telegram_messages'):
    print("Creating DataFrame...")
    df = pd.DataFrame(tqdm(messages, desc="Building DataFrame", unit="row"), columns=[
                     'channel_id', 'message_id', 'date', 'text'])
    
    print("Saving to CSV...")
    df.to_csv(f'{channel_name}.csv', index=False, 
              chunksize=1000, encoding='utf-8')
    print(f'\nMessages saved to {channel_name}.csv')


async def scrape_channel(channel, db_table):
    print(f"\nStarting scrape of {channel}")
    messages = await scrape_messages(client, channel)

    channel_name = channel.split('/')[-1]

    save_to_db(messages, db_table)
    save_to_csv(messages, channel_name)


with client:
    print("Starting Telegram scraping session...")
    client.loop.run_until_complete(scrape_channel('https://t.me/bbbreaking', 'telegram_ru'))
    print("\nScraping completed successfully!")