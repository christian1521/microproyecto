import logging
import sys
from types import FrameType
from typing import List, cast

from loguru import logger
from pydantic import AnyHttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class LoggingSettings(BaseSettings):
    LOGGING_LEVEL: int = logging.INFO


class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"

    logging: LoggingSettings = LoggingSettings()

    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "https://localhost:3000",
        "https://localhost:8000",
    ]

    PROJECT_NAME: str = "Banckchurn API"
    MODEL_DIR: str = "data/model_scibert_citing_sentences"

    model_config = SettingsConfigDict(case_sensitive=True)


class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:

        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = str(record.levelno)


        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = cast(FrameType, frame.f_back)
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level,
            record.getMessage(),
        )


def setup_app_logging(config: Settings) -> None:
    """Prepare custom logging for our application."""

    LOGGERS = ("uvicorn.asgi", "uvicorn.access")
    logging.getLogger().handlers = [InterceptHandler()]
    for logger_name in LOGGERS:
        logging_logger = logging.getLogger(logger_name)
        logging_logger.handlers = [InterceptHandler(level=config.logging.LOGGING_LEVEL)]

    logger.configure(
        handlers=[{"sink": sys.stderr, "level": config.logging.LOGGING_LEVEL}]
    )


settings = Settings()