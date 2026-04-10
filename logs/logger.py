"""
Structured logging setup via structlog.
Outputs JSON lines to file + human-readable to console.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path

import structlog


def setup_logging(log_level: str = "INFO", log_dir: str = "logs"):
    Path(log_dir).mkdir(exist_ok=True)
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Root handler: rotating file (JSON)
    file_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(log_dir, "vader.log"),
        maxBytes=10 * 1024 * 1024,   # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(level)

    # Console handler: human readable
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    logging.basicConfig(
        format="%(message)s",
        handlers=[file_handler, console_handler],
        level=level,
    )

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
