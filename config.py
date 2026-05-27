from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Tavily
    crawl_chunk_size: int = 10
    target_url: str = "https://python.langchain.com/"

    # Embeddings
    embedding_model: str = "qwen3-embedding:0.6b"
    index_batch_size: int = 40

    # Pinecone
    pinecone_api_key: str
    pinecone_index_name: str

    # LLM
    model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    groq_api_key: str

    # Langsmith settings
    langsmith_api_key: str
    langsmith_tracing: bool = True
    langsmith_project: str
    langsmith_endpoint: str = "https://api.smith.langsmith.com"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


config = Settings()
