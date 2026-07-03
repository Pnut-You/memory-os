"""FastAPI entry point for the Memory OS debug UI."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from importlib.util import find_spec
from pathlib import Path
from typing import Any, AsyncIterator

try:
    from fastapi import FastAPI, HTTPException, Path as FastAPIPath, Query, Request
    from fastapi.responses import HTMLResponse, Response
    from pydantic import BaseModel, Field
except ModuleNotFoundError:
    import re

    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str) -> None:
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class HTMLResponse(str):
        pass

    class Response(str):
        def __new__(cls, content="", media_type=None, status_code=200):  # pragma: no cover
            obj = str.__new__(cls, content)
            obj.media_type = media_type
            obj.status_code = status_code
            return obj

    class Request:  # pragma: no cover - import fallback only
        pass

    def FastAPI(*args, **kwargs):  # pragma: no cover - import fallback only
        class _App:
            def get(self, *a, **k):
                return lambda fn: fn

            def post(self, *a, **k):
                return lambda fn: fn

            def delete(self, *a, **k):
                return lambda fn: fn

        return _App()

    def FastAPIPath(*args, **kwargs):
        return None

    def Query(*args, **kwargs):
        return kwargs.get("default")

    class Field:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class BaseModel:
        def __init__(self, **data) -> None:
            hints = getattr(self.__class__, "__annotations__", {})
            for name, hint in hints.items():
                default = getattr(self.__class__, name, None)
                if name not in data:
                    if isinstance(default, Field) and "default" in default.kwargs:
                        value = default.kwargs["default"]
                    elif not isinstance(default, Field):
                        value = default
                    else:
                        raise ValueError(f"{name} is required")
                else:
                    value = data[name]
                field = getattr(self.__class__, name, None)
                if isinstance(field, Field):
                    min_length = field.kwargs.get("min_length")
                    max_length = field.kwargs.get("max_length")
                    pattern = field.kwargs.get("pattern")
                    if min_length and len(value) < min_length:
                        raise ValueError(f"{name} is too short")
                    if max_length and len(value) > max_length:
                        raise ValueError(f"{name} is too long")
                    if pattern and not re.match(pattern + "$", value):
                        raise ValueError(f"{name} is invalid")
                setattr(self, name, value)

from memory import MemoryConfig, MemoryManager

from .llm import DebugChatLLM
from .router import MemoryDebugRouter


UI_DIR = Path(__file__).resolve().parent
INDEX_FILE = UI_DIR / "static" / "index.html"
FAVICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><rect width="64" height="64" rx="14" fill="#2563eb"/><path d="M18 44V20h7l7 14 7-14h7v24h-6V30l-6 12h-4l-6-12v14z" fill="#fff"/></svg>"""
logger = logging.getLogger("memory_ui")


def _configure_logging() -> None:
    level_name = os.getenv("MEMORY_UI_LOG_LEVEL", "DEBUG").upper()
    level = getattr(logging, level_name, logging.DEBUG)
    for configured_logger in (logger, logging.getLogger("memory")):
        configured_logger.setLevel(level)
        if not configured_logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
            )
            configured_logger.addHandler(handler)
        configured_logger.propagate = False


class QueryRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    device_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    query: str = Field(min_length=1, max_length=10_000)
    debug: bool = False


class DeviceStateRequest(BaseModel):
    state: dict[str, Any]
    observed_at: str | None = None


class TimeMemoryRequest(BaseModel):
    device_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    memory_date: str = Field(min_length=1, max_length=32)


class EventSummaryRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    device_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    summary: str = Field(min_length=1, max_length=10_000)
    event_at: str = Field(min_length=1, max_length=128)
    title: str = Field(default="", max_length=200)
    metadata: dict[str, Any] = {}


class PreferenceExtractRequest(BaseModel):
    device_id: str | None = Field(default=None, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    force: bool = True
    recent_user_messages: int = 20


class WeeklyActionPreferenceRequest(BaseModel):
    device_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    end_date: str = Field(min_length=1, max_length=32)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    _configure_logging()
    app.state.memory_manager = None
    app.state.debug_router = None
    app.state.initialization_error = None
    try:
        if find_spec("redis") is None:
            raise RuntimeError(
                "Redis server may be running, but the Python 'redis' client is missing from "
                "this UI environment. Install dependencies with: uv sync"
            )
        config = MemoryConfig.from_env()
        manager = MemoryManager.create(config, start_scheduler=True)
        llm = DebugChatLLM(
            config.llm_api_key,
            config.llm_base_url,
            config.llm_model,
            config.llm_api_key_source,
        )
        app.state.memory_manager = manager
        app.state.debug_router = MemoryDebugRouter(manager, llm)
        logger.debug(
            "ui.start redis_backend=%s sqlite_path=%s llm_model=%s",
            manager.redis.backend,
            manager.events.path,
            config.llm_model,
        )
    except Exception as exc:
        app.state.initialization_error = str(exc)
        logger.exception("ui.initialization_failed")
    try:
        yield
    finally:
        manager = app.state.memory_manager
        if manager is not None:
            manager.close()
        logger.debug("ui.stop")


app = FastAPI(title="Memory OS Debug UI", version="1.0.0", lifespan=lifespan)


def _router_or_503(request: Request) -> MemoryDebugRouter:
    router = request.app.state.debug_router
    if router is None:
        detail = request.app.state.initialization_error or "MemoryManager is not initialized"
        raise HTTPException(status_code=503, detail=detail)
    return router


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(INDEX_FILE.read_text(encoding="utf-8"))


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(content=FAVICON_SVG, media_type="image/svg+xml")


@app.get("/api/status")
def status(request: Request) -> dict:
    router = request.app.state.debug_router
    if router is None:
        return {
            "ready": False,
            "error": request.app.state.initialization_error or "MemoryManager is not initialized",
        }
    return router.status()


@app.post("/api/query")
def submit_query(payload: QueryRequest, request: Request) -> dict:
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="query must not be blank")
    try:
        return _router_or_503(request).submit(payload.user_id, payload.device_id, query, payload.debug)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/api/debug/users/{user_id}")
def debug_user(
    request: Request,
    user_id: str = FastAPIPath(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"),
) -> dict:
    return _router_or_503(request).debug_user(user_id)


@app.get("/api/debug/users/{user_id}/preferences")
def debug_user_preferences(
    request: Request,
    user_id: str = FastAPIPath(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"),
) -> dict:
    return _router_or_503(request).preferences(user_id)


@app.get("/api/debug/users/{user_id}/sessions")
def debug_user_sessions(
    request: Request,
    user_id: str = FastAPIPath(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"),
    device_id: str | None = Query(default=None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"),
    local_date: str | None = Query(default=None, min_length=1, max_length=32),
) -> dict:
    return _router_or_503(request).sessions(user_id, device_id, local_date)


@app.get("/api/debug/users/{user_id}/sessions/{session_id}")
def debug_user_session_detail(
    request: Request,
    user_id: str = FastAPIPath(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"),
    session_id: str = FastAPIPath(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"),
) -> dict:
    return _router_or_503(request).session_detail(user_id, session_id)


@app.post("/api/debug/users/{user_id}/preferences/extract")
def extract_debug_user_preferences(
    payload: PreferenceExtractRequest,
    request: Request,
    user_id: str = FastAPIPath(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"),
) -> dict:
    return _router_or_503(request).extract_user_preferences(
        user_id,
        payload.device_id,
        force=payload.force,
        recent_user_messages=payload.recent_user_messages,
    )


@app.post("/api/debug/users/{user_id}/actions/preferences/extract")
def extract_debug_weekly_action_preferences(
    payload: WeeklyActionPreferenceRequest,
    request: Request,
    user_id: str = FastAPIPath(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"),
) -> dict:
    return _router_or_503(request).extract_weekly_action_preferences(
        user_id,
        payload.device_id,
        payload.end_date.strip(),
    )


@app.get("/api/debug/users/{user_id}/events")
def debug_user_events(
    request: Request,
    user_id: str = FastAPIPath(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"),
    device_id: str | None = Query(default=None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"),
    role: str | None = Query(default=None),
) -> dict:
    return _router_or_503(request).events(user_id, device_id, role)


@app.get("/api/debug/events")
def debug_event_library(
    request: Request,
    user_id: str | None = Query(default=None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"),
    device_id: str | None = Query(default=None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"),
    event_type: str | None = Query(default=None),
    session_id: str | None = Query(default=None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"),
) -> dict:
    return _router_or_503(request).event_library(user_id, device_id, event_type, session_id)


@app.get("/api/memories/events-text")
def event_text_memories(
    request: Request,
    user_id: str | None = Query(default=None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"),
    device_id: str | None = Query(default=None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"),
    event_type: str | None = Query(default="action_memory"),
    session_id: str | None = Query(default=None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"),
    memory_date: str | None = Query(default=None, min_length=1, max_length=32),
) -> dict:
    return _router_or_503(request).event_texts(user_id, device_id, event_type, session_id, memory_date)


@app.post("/api/debug/events/summaries")
def create_debug_event_summary(
    payload: EventSummaryRequest,
    request: Request,
) -> dict:
    return _router_or_503(request).create_event_summary(
        payload.user_id,
        payload.device_id,
        payload.summary.strip(),
        payload.event_at.strip(),
        payload.title.strip(),
        payload.metadata,
    )


@app.get("/api/debug/users/{user_id}/actions")
def debug_user_actions(
    request: Request,
    user_id: str = FastAPIPath(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"),
    device_id: str | None = Query(default=None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"),
) -> dict:
    return _router_or_503(request).action_events(user_id, device_id)


@app.get("/api/debug/users/{user_id}/time-memories")
def debug_user_time_memories(
    request: Request,
    user_id: str = FastAPIPath(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"),
    device_id: str | None = Query(default=None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"),
) -> dict:
    return _router_or_503(request).time_memories(user_id, device_id)


@app.post("/api/debug/users/{user_id}/time-memories")
def create_debug_time_memory(
    payload: TimeMemoryRequest,
    request: Request,
    user_id: str = FastAPIPath(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"),
) -> dict:
    return _router_or_503(request).extract_daily_memory(
        user_id,
        payload.device_id,
        payload.memory_date.strip(),
    )


@app.get("/api/debug/devices/{device_id}")
def debug_device(
    request: Request,
    device_id: str = FastAPIPath(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"),
) -> dict:
    return _router_or_503(request).debug_device(device_id)


@app.post("/api/debug/devices/{device_id}/state")
def update_debug_device_state(
    payload: DeviceStateRequest,
    request: Request,
    device_id: str = FastAPIPath(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"),
) -> dict:
    return _router_or_503(request).update_debug_device_state(
        device_id,
        payload.state,
        payload.observed_at,
    )


@app.post("/api/debug/memory-jobs/process")
def process_debug_memory_jobs(request: Request) -> dict:
    return _router_or_503(request).process_memory_jobs()


@app.delete("/api/debug/users/{user_id}/memory")
def delete_user_memory(
    request: Request,
    user_id: str = FastAPIPath(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"),
) -> dict:
    return _router_or_503(request).delete_user_memory(user_id)
