#!/usr/bin/env python3
"""
Web server entry point.
Run locally:  python server.py
Run prod:     uvicorn server:app --host 0.0.0.0 --port 8000
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))

from web.app import app  # noqa: F401 — exported for uvicorn

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        reload=os.environ.get("DEV", "false").lower() == "true",
        log_level="info",
    )
