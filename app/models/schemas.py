from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Set


class BeliefState(BaseModel):
    """
    The authoritative typed belief state for a single game session.
    All fields are updated server-side after every turn.
    """
    turn: int = 0
    confirmed: Dict[str, bool] = Field(default_factory=dict)
    candidate_scores: Dict[str, float] = Field(default_factory=dict)
    eliminated_animals: List[str] = Field(default_factory=list)
    asked_questions: List[str] = Field(default_factory=list)
    last_question: Optional[str] = None
    confidence: float = 0.0
    top_candidate: Optional[str] = None
    consecutive_unknowns: int = 0


class AnswerRequest(BaseModel):
    user_answer: str


class LLMTurnOutput(BaseModel):
    """
    Strict contract the LLM must return each turn.
    Parsed and validated by the server — never trusted blindly.
    """
    reasoning: str
    new_traits: Dict[str, bool] = Field(default_factory=dict)
    eliminated_animals: List[str] = Field(default_factory=list)
    action: str
    question: Optional[str] = None
    guess: Optional[str] = None
    confidence: float = 0.0
    candidates_remaining: int = 0


class ReasoningLog(BaseModel):
    analysis: str
    strategy: str
    confidence: float
    candidates_remaining: int


class GameResponse(BaseModel):
    action: str
    question: Optional[str] = None
    guess: Optional[str] = None
    reasoning: Optional[ReasoningLog] = None
    turn: int = 0


class StartResponse(BaseModel):
    session_id: str
    action: str
    question: str
    reasoning: Optional[ReasoningLog] = None
    turn: int = 0
