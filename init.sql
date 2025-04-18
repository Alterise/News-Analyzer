CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS embeddings (
    id BIGSERIAL PRIMARY KEY,
    special_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    input_text TEXT NOT NULL,
    embedding_vector VECTOR,
    embedding_size INTEGER,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (model_name, special_id)
);

CREATE TABLE IF NOT EXISTS embeddings_fin (
    id BIGSERIAL PRIMARY KEY,
    special_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    input_text TEXT NOT NULL,
    embedding_vector VECTOR,
    embedding_size INTEGER,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (model_name, special_id)
);

CREATE TABLE IF NOT EXISTS telegram_raw (
    id BIGSERIAL PRIMARY KEY,
    channel_id INT,
    message_id INT,
    date TIMESTAMP,
    text TEXT,
    UNIQUE (channel_id, message_id)
);

CREATE TABLE IF NOT EXISTS telegram_ru (
    id BIGSERIAL PRIMARY KEY,
    channel_id BIGINT,
    message_id INTEGER,
    date TIMESTAMP,
    text TEXT,
    UNIQUE(channel_id, message_id)
);

CREATE TABLE IF NOT EXISTS telegram_en (
    id BIGSERIAL PRIMARY KEY,
    channel_id BIGINT,
    message_id INTEGER,
    date TIMESTAMP,
    text TEXT,
    UNIQUE(channel_id, message_id)
);
