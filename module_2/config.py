from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """
    Configuration for the Triway embedding module.
    """

    embedding_provider: str = Field(
        default="local",
        alias="EMBEDDING_PROVIDER",
    )

    local_embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="LOCAL_EMBEDDING_MODEL",
    )

    local_embedding_batch_size: int = Field(
        default=16,
        alias="LOCAL_EMBEDDING_BATCH_SIZE",
    )

    normalize_embeddings: bool = Field(
        default=True,
        alias="NORMALIZE_EMBEDDINGS",
    )

    knowledge_base_path: Path = Field(
        default=Path("data/triway_knowledge_base_v0_2.json"),
        alias="KNOWLEDGE_BASE_PATH",
    )

    output_directory: Path = Field(
        default=Path("data/embeddings"),
        alias="OUTPUT_DIRECTORY",
    )

    embedding_version: str = Field(
        default="triway-services-local-v1",
        alias="EMBEDDING_VERSION",
    )

    app_env: str = Field(
        default="development",
        alias="APP_ENV",
    )

    log_level: str = Field(
        default="INFO",
        alias="LOG_LEVEL",
    )

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("embedding_provider")
    @classmethod
    def validate_provider(cls, value: str) -> str:
        value = value.strip().lower()

        allowed = {"local", "openai", "azure_openai"}

        if value not in allowed:
            raise ValueError(
                f"EMBEDDING_PROVIDER must be one of: {sorted(allowed)}"
            )

        return value

    @field_validator("local_embedding_batch_size")
    @classmethod
    def validate_batch_size(cls, value: int) -> int:
        if value < 1:
            raise ValueError(
                "LOCAL_EMBEDDING_BATCH_SIZE must be at least 1."
            )

        return value

    @field_validator("knowledge_base_path")
    @classmethod
    def resolve_knowledge_base_path(cls, value: Path) -> Path:
        if value.is_absolute():
            return value

        return PROJECT_ROOT / value

    @field_validator("output_directory")
    @classmethod
    def resolve_output_directory(cls, value: Path) -> Path:
        if value.is_absolute():
            return value

        return PROJECT_ROOT / value


settings = Settings()