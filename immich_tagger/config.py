"""
Configuration management for the Immich Auto-Tagger service.
"""

import json
from typing import Dict, List
from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with validation."""
    
    # Immich Configuration
    immich_base_url: str = Field(..., env="IMMICH_BASE_URL")
    immich_api_key: str = Field(default="", env="IMMICH_API_KEY")  # Legacy single key support
    immich_api_keys: List[str] = Field(default=[], env="IMMICH_API_KEYS")  # New multi-key support
    immich_libraries: Dict[str, str] = Field(default={}, env="IMMICH_LIBRARIES")  # Named libraries
    
    # Processing Configuration
    confidence_threshold: float = Field(default=0.35, env="CONFIDENCE_THRESHOLD", ge=0.0, le=1.0)
    batch_size: int = Field(default=250, env="BATCH_SIZE", gt=0, description="Maximum assets processed per batch")
    max_batches_per_run: int = Field(
        default=0,
        env="MAX_BATCHES_PER_RUN",
        ge=0,
        description="Maximum batches processed per scheduled run (0 = unlimited)",
    )
    processed_tag_name: str = Field(default="auto:processed", env="PROCESSED_TAG_NAME")
    content_rating_tag_name: str = Field(default="content-rating", env="CONTENT_RATING_TAG_NAME")
    target_albums: str = Field(default="", env="TARGET_ALBUMS")
    failure_timeout: int = Field(default=3, env="FAILURE_TIMEOUT", ge=0, description="Max retries for failed assets (0 = never retry)")
    config_dir: str = Field(default="/config", env="CONFIG_DIR")
    
    # Model Configuration
    tagging_model: str = Field(default="wd14", env="TAGGING_MODEL")
    wd_model_repo: str = Field(
        default="SmilingWolf/wd-swinv2-tagger-v3",
        env="WD_MODEL_REPO",
    )
    model_cache_dir: str = Field(default="/config/models", env="MODEL_CACHE_DIR")
    inference_batch_size: int = Field(
        default=8,
        env="INFERENCE_BATCH_SIZE",
        gt=0,
        description="Maximum images passed to ONNX Runtime at once",
    )
    character_threshold: float = Field(
        default=0.9,
        env="CHARACTER_THRESHOLD",
        ge=0.0,
        le=1.0,
    )
    onnx_intra_op_threads: int = Field(
        default=0,
        env="ONNX_INTRA_OP_THREADS",
        ge=0,
        description="ONNX worker threads (0 lets ONNX Runtime decide)",
    )
    unload_model_after_run: bool = Field(
        default=False,
        env="UNLOAD_MODEL_AFTER_RUN",
        description="Release the ONNX model between distant scheduled runs",
    )
    
    # Performance Configuration
    max_retries: int = Field(default=3, env="MAX_RETRIES", gt=0)
    retry_delay: float = Field(default=1.0, env="RETRY_DELAY", gt=0.0)
    request_timeout: float = Field(default=30.0, env="REQUEST_TIMEOUT", gt=0.0)
    tag_cache_ttl: int = Field(default=300, env="TAG_CACHE_TTL", gt=0)  # Tag cache TTL in seconds
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Health endpoint
    health_port: int = Field(default=8000, env="HEALTH_PORT")
    
    # Scheduling Configuration
    enable_scheduler: bool = Field(default=True, env="ENABLE_SCHEDULER")
    cron_schedule: str = Field(default="0 2 * * *", env="CRON_SCHEDULE")  # Daily at 2 AM
    timezone: str = Field(default="UTC", env="TIMEZONE")
    
    @validator("immich_base_url")
    def validate_immich_url(cls, v):
        """Ensure the Immich URL is properly formatted."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("IMMICH_BASE_URL must start with http:// or https://")
        return v.rstrip("/")
    
    @validator("tagging_model")
    def validate_tagging_model(cls, v):
        """Ensure the tagging model is supported."""
        supported_models = ["wd14"]
        if v.lower() not in supported_models:
            raise ValueError(f"TAGGING_MODEL must be one of: {supported_models}")
        return v.lower()

    @validator("content_rating_tag_name")
    def validate_content_rating_tag_name(cls, v):
        """Ensure the content-rating parent is a single, usable tag name."""
        value = v.strip()
        if not value:
            raise ValueError("CONTENT_RATING_TAG_NAME cannot be empty")
        if "/" in value or any(char in value for char in ("\n", "\r", "\t")):
            raise ValueError(
                "CONTENT_RATING_TAG_NAME must be a single tag name without '/'"
            )
        return value
    
    @validator("log_level")
    def validate_log_level(cls, v):
        """Ensure the log level is valid."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of: {valid_levels}")
        return v.upper()
    
    @validator('immich_api_keys', pre=True, always=True)
    def parse_api_keys(cls, v, values):
        """Parse API keys from various formats and ensure we have at least one."""
        if isinstance(v, str):
            if not v:  # Empty string, try legacy single key
                single_key = values.get('immich_api_key', '')
                return [single_key] if single_key else []
            # Support JSON array format
            if v.startswith('[') and v.endswith(']'):
                try:
                    return json.loads(v)
                except json.JSONDecodeError:
                    raise ValueError("Invalid JSON format for IMMICH_API_KEYS")
            else:
                # Support comma-separated format
                return [key.strip() for key in v.split(',') if key.strip()]
        elif isinstance(v, list):
            return v
        else:
            # No multi-keys provided, use legacy single key
            single_key = values.get('immich_api_key', '')
            return [single_key] if single_key else []
    
    @validator('immich_libraries', pre=True)
    def parse_libraries(cls, v):
        """Parse named libraries from JSON format."""
        if isinstance(v, str):
            if not v:  # Empty string
                return {}
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON format for IMMICH_LIBRARIES")
        return v if v else {}
    
    def get_library_names(self) -> List[str]:
        """Get library names (or generate default names)."""
        if self.immich_libraries:
            return list(self.immich_libraries.keys())
        else:
            # Generate default names for API keys
            return [f"Library_{i+1}" for i in range(len(self.immich_api_keys))]
    
    def get_api_keys(self) -> List[str]:
        """Get all API keys."""
        if self.immich_libraries:
            return list(self.immich_libraries.values())
        return self.immich_api_keys
    
    def get_library_config(self) -> List[Dict[str, str]]:
        """Get library configuration as list of {name, api_key} dicts."""
        if self.immich_libraries:
            return [{"name": name, "api_key": key} for name, key in self.immich_libraries.items()]
        else:
            api_keys = self.get_api_keys()
            return [{"name": f"Library_{i+1}", "api_key": key} for i, key in enumerate(api_keys)]
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
