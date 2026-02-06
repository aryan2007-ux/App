from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

MemoryType = Literal["preference", "constraint", "instruction", "commitment", "profile", "entity"]
OpAction = Literal["create", "update", "invalidate", "noop"]


class ChatTurnIn(BaseModel):
    user_id: str
    conversation_id: str = "default"
    turn_id: int
    message: str


class MemoryOut(BaseModel):
    memory_id: str
    type: MemoryType
    key: str
    value: Any
    value_text: str
    confidence: float
    salience: float
    source_turn_id: Optional[int] = None
    source_quote: Optional[str] = None
    last_used_turn_id: Optional[int] = None
    times_used: int = 0


class RetrieveResponse(BaseModel):
    active_memories: list[MemoryOut]
    memory_context: str
    latency_ms: dict[str, float] = Field(default_factory=dict)


class ExtractOp(BaseModel):
    action: OpAction
    type: MemoryType
    key: str
    value: Any = Field(default_factory=dict)
    value_text: str = ""
    confidence: float = 0.0
    salience: float = 0.5
    ttl_hours: int = 0
    source_quote: str = ""


class ExtractResponse(BaseModel):
    ops: list[ExtractOp] = Field(default_factory=list)
