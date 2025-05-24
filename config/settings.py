from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    openai_api_key: str
    openai_model: str = "gpt-4.1"
    openai_embedding_model: str = "text-embedding-3-large"

    app_name: str = "SemanticScout"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"

    chroma_persist_dir: Path = Path("./data/chroma_db")
    upload_dir: Path = Path("./data/uploads")
    max_file_size: int = 104_857_600  # 100MB
    supported_formats: str = "pdf,docx,txt,md"

    gradio_theme: str = "soft"
    gradio_share: bool = False
    gradio_port: int = 7860
    gradio_server_name: str = "127.0.0.1"

    embedding_dimensions: int = 3072
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_documents: int = 1000

    dev_mode: bool = False
    enable_logging: bool = True
    log_file: Path = Path("./logs/semantic_scout.log")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Return a cached Settings instance."""

    return Settings()
