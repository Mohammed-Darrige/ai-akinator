"""Schemas for the LLM-native Akinator backend."""

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class ConstraintLedger(BaseModel):
    """Server-owned record of confirmed facts and resolved answers."""

    facts: Dict[str, bool] = Field(default_factory=dict)
    qa_history: List[Tuple[str, str]] = Field(default_factory=list)


class GameState(BaseModel):
    """Mutable session state managed by the backend."""

    language: str = "tr"
    turn: int = 0
    ledger: ConstraintLedger = Field(default_factory=ConstraintLedger)
    conversation: List[Dict[str, str]] = Field(default_factory=list)
    asked_traits: List[str] = Field(default_factory=list)
    trait_labels: Dict[str, str] = Field(default_factory=dict)
    confidence: float = 0.0
    top_candidates: List[Dict[str, Any]] = Field(default_factory=list)
    pending_guess: Optional[str] = None
    last_trait_key: Optional[str] = None
    last_question: Optional[str] = None
    game_over: bool = False


class StartRequest(BaseModel):
    lang: str = "tr"
    old_session_id: Optional[str] = None


class AnswerRequest(BaseModel):
    user_answer: str
    lang: str = "tr"


class ReasoningLog(BaseModel):
    """Reasoning payload sent back with each completed turn."""

    chain_of_thought: str = ""
    confidence: float = 0.0
    candidates_remaining: int = 0
    constraints: Dict[str, bool] = Field(default_factory=dict)
    qa_history: List[Tuple[str, str]] = Field(default_factory=list)
    trait_labels: Dict[str, str] = Field(default_factory=dict)
    top_candidates: List[Dict[str, Any]] = Field(default_factory=list)
    has_contradiction: bool = False
