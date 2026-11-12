from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    # Database
    database_url: str = "sqlite+aiosqlite:///./data/afd.db"

    # Paths
    bundled_dataset_dir: str = "data/CIC-IDS2017"
    model_artifact_dir: str = "models"

    # AE fine-tune
    fine_tune_k_threshold: int = 50
    ae_anomaly_threshold_percentile: float = 95.0
    ae_replay_buffer_fraction: float = 0.10
    ae_freeze_epochs: int = 5

    # Device — "auto" resolves at startup via torch.cuda.is_available()
    device: str = "auto"

    # Regression gates
    far_gate_max: float = 0.05
    recall_gate_max: float = 0.05

    # Feature engineering
    smote_max_per_class: int = 50_000
    variance_threshold: float = 0.0
    correlation_threshold: float = 0.95
    feature_selection_k: int = 0
    feature_selection_method: str = "mi"

    # Inference
    clf_confidence_threshold: float = 0.5
    max_upload_mb: int = 100
    sync_max_rows: int = 5_000

    # API
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost"]
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
