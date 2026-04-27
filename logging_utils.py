from __future__ import annotations

import logging


def configure_logging(level: int = logging.INFO) -> None:
    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.setLevel(level)
        return

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    for noisy_logger in (
        "httpx",
        "huggingface_hub",
        "sentence_transformers",
        "transformers",
        "arxiv",
    ):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
