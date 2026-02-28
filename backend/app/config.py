"""Application configuration via environment variables."""
from pathlib import Path
from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    app_name: str = "VisDL"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    upload_dir: Path = PROJECT_ROOT / "data" / "uploads"
    weights_dir: Path = PROJECT_ROOT / "data" / "weights"
    training_logs_dir: Path = PROJECT_ROOT / "data" / "training_logs"
    graphs_dir: Path = PROJECT_ROOT / "data" / "graphs"
    max_upload_size_mb: int = 4000
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    model_config = {"env_prefix": "VISDL_"}


settings = Settings()
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.weights_dir.mkdir(parents=True, exist_ok=True)
settings.training_logs_dir.mkdir(parents=True, exist_ok=True)
settings.graphs_dir.mkdir(parents=True, exist_ok=True)
