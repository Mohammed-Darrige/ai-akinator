import json
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from app.models.schemas import AnswerRequest, StartRequest
from app.services.llm_service import clear_session, process_turn_stream

router = APIRouter()

_SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _sse_response(generator: AsyncGenerator[str, None]) -> StreamingResponse:
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


def _session_event(session_id: str) -> str:
    payload = {"type": "session_id", "session_id": session_id}
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("/start")
async def start_game(request: StartRequest):
    if request.old_session_id:
        clear_session(request.old_session_id)

    session_id = str(uuid.uuid4())

    async def stream_generator() -> AsyncGenerator[str, None]:
        yield _session_event(session_id)
        async for chunk in process_turn_stream(session_id, language=request.lang):
            yield chunk

    return _sse_response(stream_generator())


@router.post("/ask")
async def ask_question(request: AnswerRequest, session_id: str = Query(...)):
    return _sse_response(
        process_turn_stream(
            session_id,
            user_answer=request.user_answer,
            language=request.lang,
        )
    )


@router.post("/restart")
async def restart_game(session_id: str = Query(...)):
    clear_session(session_id)
    return {"status": "ok"}
