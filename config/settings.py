"""Application settings using Pydantic BaseSettings."""
from __future__ import annotations

from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import model_validator, Field


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    openai_api_key: str
    openai_model: str = "gpt-4.1"
    openai_embedding_model: str = "text-embedding-3-large"
    chroma_persist_dir: str = "./data/chroma_db"
    upload_dir: str = "./data/uploads"
    max_file_size: int = 100 * 1024 * 1024
    supported_formats: List[str] = Field(default_factory=lambda: ["pdf", "docx", "txt", "md"])
    gradio_theme: str = "soft"
    gradio_share: bool = False
    gradio_port: int = 7860
    gradio_server_name: str = "127.0.0.1"
    log_level: str = "INFO"
    log_file: str = "./logs/semantic_scout.log"
    embedding_dimensions: int = 3072
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_documents: int = 1000
    app_name: str = "SemanticScout"
    app_version: str = "1.0.0"
    debug: bool = False
    dev_mode: bool = False
    enable_logging: bool = True

    @model_validator(mode="before")
    @classmethod
    def parse_supported_formats(cls, values: dict) -> dict:
        sf = values.get("supported_formats")
        if isinstance(sf, str):
            values["supported_formats"] = [i.strip() for i in sf.split(",") if i.strip()]
        return values

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

# Lazy settings instance used by application entry points.
settings: Settings | None = None

def get_settings() -> Settings:
    """Return a cached settings instance."""
    global settings
    if settings is None:
        settings = Settings()
    return settings
