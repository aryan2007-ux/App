from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "LongFormMemory"

    database_url: str = "postgresql://postgres:postgres@localhost:5432/memory"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3:8b"

    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_dim: int = 384

    max_memory_items: int = 8
    vector_top_k: int = 30

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
