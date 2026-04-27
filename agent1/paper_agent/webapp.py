from __future__ import annotations

import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import load_config
from .pipeline import DailyResearchAgent


BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "web" / "templates"
STATIC_DIR = BASE_DIR / "web" / "static"

app = FastAPI(
    title="Paper Research Agent",
    description="LangGraph-based multi-agent literature intelligence demo",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@lru_cache(maxsize=1)
def get_agent() -> DailyResearchAgent:
    return DailyResearchAgent(load_config(None))


def _json_error(message: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse({"ok": False, "error": message}, status_code=status_code)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    config = get_agent().config
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "project_name": "Paper Research Agent",
            "topic": config.topic,
            "default_thread_id": config.graph.default_thread_id,
            "default_user_id": config.graph.default_user_id,
        },
    )


@app.get("/api/health")
async def health() -> dict[str, Any]:
    agent = get_agent()
    return {
        "ok": True,
        "topic": agent.config.topic,
        "vector_dir": agent.config.paths.vector_dir,
        "report_dir": agent.config.paths.report_dir,
    }


@app.get("/api/config")
async def config_view() -> dict[str, Any]:
    return {"ok": True, "config": get_agent().config.to_dict()}


@app.get("/api/graph")
async def graph_view() -> dict[str, Any]:
    return {"ok": True, "mermaid": get_agent().show_graph()}


@app.post("/api/ask")
async def ask(payload: dict[str, Any]) -> dict[str, Any]:
    question = str(payload.get("question", "")).strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    config = get_agent().config
    result = get_agent().ask(
        question=question,
        thread_id=str(payload.get("thread_id") or config.graph.default_thread_id),
        user_id=str(payload.get("user_id") or config.graph.default_user_id),
    )
    return {"ok": True, "result": result}


@app.post("/api/run-daily")
async def run_daily() -> dict[str, Any]:
    result = get_agent().run_daily()
    return {"ok": True, "result": result.to_dict()}


@app.post("/api/ingest-upload")
async def ingest_upload(
    file: UploadFile = File(...),
    paper_id: str = Form(default=""),
    title: str = Form(default=""),
) -> dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    agent = get_agent()
    upload_dir = Path(agent.config.paths.base_dir) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    target_path = upload_dir / file.filename

    with target_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = agent.ingest_local_pdf(
        str(target_path),
        paper_id=paper_id.strip() or None,
        title=title.strip() or None,
    )
    return {"ok": True, "result": result}


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    return _json_error(str(exc.detail), status_code=exc.status_code)


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    return _json_error(f"Unhandled server error: {exc}", status_code=500)
