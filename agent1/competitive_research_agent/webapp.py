from __future__ import annotations

import asyncio
import json
import shutil
import uuid
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from urllib.request import Request as UrlRequest
from urllib.request import urlopen

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import load_config
from .pipeline import CompetitiveResearchAgent


BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "web" / "templates"
STATIC_DIR = BASE_DIR / "web" / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_agent().warmup_models()
    yield


app = FastAPI(
    title="Industry Competitive Research Agent",
    description="LangGraph-based Agentic RAG system for industry and competitor research",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@lru_cache(maxsize=1)
def get_agent() -> CompetitiveResearchAgent:
    return CompetitiveResearchAgent(load_config(None))


def _json_error(message: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse({"ok": False, "error": message}, status_code=status_code)


def _safe_pdf_filename(url: str) -> str:
    parsed = urlparse(url)
    name = Path(unquote(parsed.path)).name
    if not name.lower().endswith(".pdf"):
        name = "downloaded_research_document.pdf"
    safe_name = "".join(char if char.isalnum() or char in ".-_" else "_" for char in name)
    return safe_name or "downloaded_research_document.pdf"


def _download_pdf(url: str, target_path: Path) -> None:
    request = UrlRequest(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=30) as response:
        content_type = response.headers.get("Content-Type", "").lower()
        if "pdf" not in content_type and not urlparse(url).path.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="URL does not appear to point to a PDF")
        with target_path.open("wb") as buffer:
            shutil.copyfileobj(response, buffer)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    config = get_agent().config
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "project_name": "Industry Competitive Research Agent",
            "topic": config.topic,
            "default_thread_id": uuid.uuid4().hex,
            "default_user_id": uuid.uuid4().hex,
        },
    )


@app.get("/api/health")
async def health() -> dict[str, Any]:
    agent = get_agent()
    return {
        "ok": True,
        "topic": agent.config.topic,
        "research_collection": agent.config.rag.collection_name,
        "report_dir": agent.config.paths.report_dir,
    }


@app.get("/api/config")
async def config_view() -> dict[str, Any]:
    config = get_agent().config
    return {
        "ok": True,
        "config": {
            "topic": config.topic,
            "llm_model": config.rag.llm_model,
            "embedding_model": config.rag.embedding_model,
            "reranker_model": config.rag.reranker_model,
            "collection_name": config.rag.collection_name,
        },
    }


@app.get("/api/graph")
async def graph_view() -> dict[str, Any]:
    return {"ok": True, "mermaid": get_agent().show_graph()}


@app.get("/api/history/runs")
async def history_runs(limit: int = 20) -> dict[str, Any]:
    agent = get_agent()
    if not agent.trace_store:
        return {
            "ok": True,
            "database_enabled": False,
            "runs": [],
            "message": "MySQL persistence is disabled.",
        }
    return {
        "ok": True,
        "database_enabled": True,
        "runs": agent.trace_store.list_workflow_runs(limit=max(1, min(limit, 100))),
    }


@app.get("/api/history/runs/{run_id}")
async def history_run_detail(run_id: str) -> dict[str, Any]:
    agent = get_agent()
    if not agent.trace_store:
        raise HTTPException(status_code=404, detail="MySQL persistence is disabled")

    run = agent.trace_store.get_workflow_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    return {"ok": True, "run": run}


@app.get("/api/history/evals")
async def history_evals(limit: int = 20) -> dict[str, Any]:
    agent = get_agent()
    if not agent.trace_store:
        return {
            "ok": True,
            "database_enabled": False,
            "eval_runs": [],
            "message": "MySQL persistence is disabled.",
        }
    return {
        "ok": True,
        "database_enabled": True,
        "eval_runs": agent.trace_store.list_eval_runs(limit=max(1, min(limit, 100))),
    }


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


def _sse_message(event: dict[str, Any]) -> str:
    event_name = str(event.get("event", "message"))
    return f"event: {event_name}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"


@app.post("/api/ask/stream")
async def ask_stream(payload: dict[str, Any]) -> StreamingResponse:
    question = str(payload.get("question", "")).strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    agent = get_agent()
    config = agent.config
    iterator = agent.ask_stream(
        question=question,
        thread_id=str(payload.get("thread_id") or config.graph.default_thread_id),
        user_id=str(payload.get("user_id") or config.graph.default_user_id),
    )

    async def generate():
        while True:
            event = await asyncio.to_thread(next, iterator, None)
            if event is None:
                break
            yield _sse_message(event)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/generate-brief")
async def generate_brief() -> dict[str, Any]:
    result = get_agent().generate_brief()
    return {"ok": True, "result": result.to_dict()}


@app.post("/api/ingest-upload")
async def ingest_upload(
    file: UploadFile = File(...),
    document_id: str = Form(default=""),
    title: str = Form(default=""),
    industry: str = Form(default=""),
    company: str = Form(default=""),
    product_line: str = Form(default=""),
    document_type: str = Form(default=""),
) -> dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    agent = get_agent()
    upload_dir = Path(agent.config.paths.base_dir) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    target_path = upload_dir / f"{uuid.uuid4().hex}.pdf"

    with target_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = agent.ingest_document(
        str(target_path),
        document_id=document_id.strip() or None,
        title=title.strip() or None,
        industry=industry.strip(),
        company=company.strip(),
        product_line=product_line.strip(),
        document_type=document_type.strip(),
    )
    return {"ok": True, "result": result}


@app.post("/api/ingest-url")
async def ingest_url(payload: dict[str, Any]) -> dict[str, Any]:
    url = str(payload.get("url", "")).strip()
    if not url:
        raise HTTPException(status_code=400, detail="url is required")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise HTTPException(status_code=400, detail="Only http(s) PDF URLs are supported")

    agent = get_agent()
    upload_dir = Path(agent.config.paths.base_dir) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    target_path = upload_dir / f"{uuid.uuid4().hex}_{_safe_pdf_filename(url)}"
    await asyncio.to_thread(_download_pdf, url, target_path)

    document_id = str(payload.get("document_id") or "").strip()
    result = agent.ingest_document(
        str(target_path),
        document_id=document_id or None,
        title=str(payload.get("title") or "").strip() or None,
        industry=str(payload.get("industry") or "").strip(),
        company=str(payload.get("company") or "").strip(),
        product_line=str(payload.get("product_line") or "").strip(),
        document_type=str(payload.get("document_type") or "").strip(),
    )
    return {"ok": True, "result": result}


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    return _json_error(str(exc.detail), status_code=exc.status_code)


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    return _json_error("Internal server error", status_code=500)
