import uuid
from fastapi import APIRouter, HTTPException, Query
from app.models.schemas import AnswerRequest, GameResponse, StartResponse, ReasoningLog
from app.services.llm_service import process_turn, clear_session

router = APIRouter()


def _check_errors(res: dict) -> None:
    """Raise the appropriate HTTP error for error/rate_limited responses."""
    if res.get("type") == "error":
        raise HTTPException(status_code=500, detail=res.get("content"))
    if res.get("type") == "rate_limited":
        retry_after = int(res.get("retry_after", 60))
        raise HTTPException(
            status_code=503,
            detail=f"Rate limit hit — retry after {retry_after}s",
            headers={"Retry-After": str(retry_after)},
        )


@router.post("/start", response_model=StartResponse)
async def start_game(old_session_id: str = Query(None)):
    if old_session_id:
        clear_session(old_session_id)

    session_id = str(uuid.uuid4())
    res = await process_turn(session_id)
    _check_errors(res)

    raw_reasoning = res.get("reasoning")
    reasoning = ReasoningLog(**raw_reasoning) if isinstance(raw_reasoning, dict) else None

    return StartResponse(
        session_id=session_id,
        action=res.get("action", "ask_question"),
        question=res.get("question", ""),
        reasoning=reasoning,
        turn=res.get("turn", 0),
    )


@router.post("/ask", response_model=GameResponse)
async def ask_question(request: AnswerRequest, session_id: str = Query(...)):
    res = await process_turn(session_id, request.user_answer)
    _check_errors(res)

    raw_reasoning = res.get("reasoning")
    reasoning = ReasoningLog(**raw_reasoning) if isinstance(raw_reasoning, dict) else None

    return GameResponse(
        action=res.get("action", "ask_question"),
        question=res.get("question"),
        guess=res.get("guess"),
        reasoning=reasoning,
        turn=res.get("turn", 0),
    )


@router.post("/restart")
async def restart_game(session_id: str = Query(...)):
    clear_session(session_id)
    return {"status": "ok"}
