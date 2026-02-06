CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS memories (
  memory_id uuid PRIMARY KEY,
  user_id text NOT NULL,

  type text NOT NULL,
  key text NOT NULL,

  value jsonb NOT NULL,
  value_text text NOT NULL,

  confidence real NOT NULL DEFAULT 0.0,
  salience real NOT NULL DEFAULT 0.5,

  status text NOT NULL DEFAULT 'active', -- active|superseded|invalidated

  source_turn_id bigint,
  source_quote text,

  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),

  last_used_turn_id bigint,
  times_used int NOT NULL DEFAULT 0,

  expires_at timestamptz,

  embedding vector(384),
  embedding_model text
);

CREATE INDEX IF NOT EXISTS idx_mem_user_status ON memories(user_id, status);
CREATE INDEX IF NOT EXISTS idx_mem_user_key ON memories(user_id, key, status);
CREATE INDEX IF NOT EXISTS idx_mem_user_type_key ON memories(user_id, type, key, status);

CREATE INDEX IF NOT EXISTS idx_mem_embedding_hnsw
ON memories USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
