from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv_file(path: Path | None = None) -> None:
    env_path = path or PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Settings:
    database_url: str
    embedding_provider: str
    embedding_model: str
    embedding_revision: str
    embedding_dimensions: int
    embedding_device: str
    embedding_batch_size: int
    model_cache_dir: str
    reranker_model: str
    reranker_device: str
    reranker_batch_size: int
    reranker_max_length: int
    cloudflare_account_id: str
    cloudflare_api_token: str
    llm_provider: str
    deepseek_api_key: str
    deepseek_base_url: str
    deepseek_model: str
    deepseek_max_tokens: int
    deepseek_input_cost_per_million: float
    deepseek_output_cost_per_million: float
    ops_log_retention_days: int

    @property
    def embedding_model_id(self) -> str:
        if self.embedding_provider == "local" and self.embedding_revision:
            return f"{self.embedding_model}@{self.embedding_revision}"
        return self.embedding_model

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv_file()
        return cls(
            database_url=os.getenv("DATABASE_URL", "sqlite:///./data/jobrag.db"),
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", "local"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
            embedding_revision=os.getenv("EMBEDDING_REVISION", "refs/pr/130"),
            embedding_dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", "1024")),
            embedding_device=os.getenv("EMBEDDING_DEVICE", "auto"),
            embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "2")),
            model_cache_dir=os.getenv("MODEL_CACHE_DIR", "./data/models"),
            reranker_model=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
            reranker_device=os.getenv("RERANKER_DEVICE", "auto"),
            reranker_batch_size=int(os.getenv("RERANKER_BATCH_SIZE", "8")),
            reranker_max_length=int(os.getenv("RERANKER_MAX_LENGTH", "256")),
            cloudflare_account_id=os.getenv("CLOUDFLARE_ACCOUNT_ID", ""),
            cloudflare_api_token=os.getenv("CLOUDFLARE_API_TOKEN", ""),
            llm_provider=os.getenv("LLM_PROVIDER", "deepseek"),
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            deepseek_max_tokens=int(os.getenv("DEEPSEEK_MAX_TOKENS", "1200")),
            deepseek_input_cost_per_million=float(os.getenv("DEEPSEEK_INPUT_COST_PER_MILLION", "0")),
            deepseek_output_cost_per_million=float(os.getenv("DEEPSEEK_OUTPUT_COST_PER_MILLION", "0")),
            ops_log_retention_days=int(os.getenv("OPS_LOG_RETENTION_DAYS", "30")),
        )


settings = Settings.from_env()
