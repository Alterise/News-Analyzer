import os
import psycopg2
import pandas as pd

from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.tl.functions.channels import JoinChannelRequest

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


async def scrape_messages(client, channel, limit=3000):
    messages = []
    async for message in client.iter_messages(channel, limit):
        if message.text:
            messages.append({
                'channel_id': message.peer_id.channel_id,
                'message_id': message.id,
                'date': message.date,
                'text': message.text
            })

    return messages


def save_to_db(messages):
    db = psycopg2.connect(host='localhost', dbname='postgres',
                          user='postgres', password='admin', port=5432)

    cursor = db.cursor()

    cursor.execute("""CREATE TABLE IF NOT EXISTS message (
        id BIGSERIAL PRIMARY KEY,
        channel_id INT,
        message_id INT,
        date TIMESTAMP,
        text TEXT
    );
    """)

    for message in messages:
        cursor.execute("INSERT INTO message (channel_id, message_id, date, text) VALUES (%s, %s, %s, %s)",
                       (message.channel_id, message.id, message.date, message.text))

    db.commit()

    cursor.close()

    db.close()

    print(f'Messages saved to db')


def save_to_csv(messages, channel_name='telegram_messages'):
    df = pd.DataFrame(messages, columns=[
                      'channel_id', 'message_id', 'date', 'text'])
    df.to_csv(f'{channel_name}.csv', index=False)
    print(f'Messages saved to {channel_name}.csv')


async def scrape_channel(channel):
    # await join_channel(client, channel)

    messages = await scrape_messages(client, channel, None)

    channel_name = channel.split('/')[-1]

    save_to_csv(messages, channel_name)

with client:
    client.loop.run_until_complete(scrape_channel('https://t.me/cbrstocks'))
