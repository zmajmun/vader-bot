"""
FastAPI application — mounts all routers and serves the web dashboard.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from web.models import init_db
from web.routers.auth_router import router as auth_router
from web.routers.bot_router import router as bot_router
from web.routers.chart_router import router as chart_router

WEB_DIR = Path(__file__).parent

app = FastAPI(
    title="VADER — SMC Trading Bot",
    description="Smart Money Concepts algorithmic trading platform",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(auth_router)
app.include_router(bot_router)
app.include_router(chart_router)

# Static files
static_dir = WEB_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.on_event("startup")
async def startup():
    init_db()


# Serve SPA for all non-API routes
@app.get("/{full_path:path}", include_in_schema=False)
async def serve_spa(full_path: str):
    index = WEB_DIR / "templates" / "index.html"
    return FileResponse(str(index))
