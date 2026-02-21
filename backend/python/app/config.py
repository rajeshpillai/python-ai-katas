from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    cors_origins: list[str] = ["http://localhost:3000"]
    sandbox_timeout_seconds: int = 30
    sandbox_memory_limit_mb: int = 512
    sandbox_cpu_limit: float = 1.0

    class Config:
        env_prefix = "KATAS_"


settings = Settings()
