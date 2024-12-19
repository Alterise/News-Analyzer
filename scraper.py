import os
import psycopg2
from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.tl.functions.channels import JoinChannelRequest

db = psycopg2.connect(host='localhost', dbname='postgres', user='postgres', password='admin', port=5432)

cursor = db.cursor()

cursor.execute("""CREATE TABLE IF NOT EXISTS message (
    id BIGSERIAL PRIMARY KEY,
    channel_id INT,
    message_id INT,
    text TEXT,
    date TIMESTAMP
);
""")

load_dotenv()
api_id = os.getenv("TELEGRAM_API_ID")
api_hash = os.getenv("TELEGRAM_API_HASH")

client = TelegramClient('news_scraping', api_id, api_hash)


async def join_channel(client, channel_link):
    try:
        await client(JoinChannelRequest(channel_link))
        print(f"Successfully joined the channel {channel_link}")
    except:
        print(f"Failed to join the channel {channel_link}")


async def scrape_message(client, channel, limit=100):
    async for message in client.iter_messages(channel, limit):
        if message.text:
            cursor.execute("INSERT INTO message (channel_id, message_id, text, date) VALUES (%s, %s, %s, %s)", 
                           (message.peer_id.channel_id, message.id, message.text, message.date))


async def main():
    channel_link = 'https://t.me/bbbreaking'
    await join_channel(client, channel_link)

    await scrape_message(client, channel_link, 50000)


with client:
    client.loop.run_until_complete(main())

db.commit()

cursor.close()

db.close()