import json
import time
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, HTTPException

from app.db import get_conn
from app.embeddings import embed_text
from app.extraction import extract_ops
from app.models import ChatTurnIn, MemoryOut, RetrieveResponse
from app.retrieval import retrieve_context
from app.settings import settings

app = FastAPI(title=settings.app_name)


def _now():
    return datetime.now(timezone.utc)


def _ttl_to_expires(ttl_hours: int):
    if ttl_hours and ttl_hours > 0:
        return _now() + timedelta(hours=ttl_hours)
    return None


@app.get("/health")
def health():
    return {"ok": True, "app": settings.app_name}


@app.post("/turn", response_model=RetrieveResponse)
def handle_turn(turn: ChatTurnIn):
    overall_t0 = time.perf_counter()

    # 1) Retrieve relevant context
    active, context, latency_retr = retrieve_context(turn.user_id, turn.message)

    # 2) Ask LLM to produce memory ops
    extract_resp, latency_ext = extract_ops(turn.user_id, turn.turn_id, turn.message, active)

    # 3) Apply ops
    applied = 0
    with get_conn() as conn:
        for op in extract_resp.ops:
            if op.action == "noop":
                continue

            if op.action == "invalidate":
                conn.execute(
                    """
                    UPDATE memories
                    SET status='invalidated', updated_at=now()
                    WHERE user_id=%s AND status='active' AND type=%s AND key=%s
                    """,
                    (turn.user_id, op.type, op.key),
                )
                applied += 1
                continue

            mem_id = str(uuid.uuid4())
            expires_at = _ttl_to_expires(op.ttl_hours)
            emb = embed_text(op.value_text)

            conn.execute(
                """
                INSERT INTO memories (
                    memory_id, user_id, type, key, value, value_text,
                    confidence, salience, status, source_turn_id, source_quote,
                    created_at, updated_at, expires_at,
                    embedding, embedding_model
                )
                VALUES (
                    %s, %s, %s, %s, %s::jsonb, %s,
                    %s, %s, 'active', %s, %s,
                    now(), now(), %s,
                    %s::vector, %s
                )
                """,
                (
                    mem_id,
                    turn.user_id,
                    op.type,
                    op.key,
                    json.dumps(op.value),
                    op.value_text,
                    op.confidence,
                    op.salience,
                    turn.turn_id,
                    op.source_quote,
                    expires_at,
                    emb,
                    settings.embed_model,
                ),
            )
            applied += 1

        # expire old
        conn.execute(
            """
            UPDATE memories
            SET status='invalidated', updated_at=now()
            WHERE user_id=%s AND status='active' AND expires_at IS NOT NULL AND expires_at < now()
            """,
            (turn.user_id,),
        )

    # 4) Return updated context
    active2, context2, latency_retr2 = retrieve_context(turn.user_id, turn.message)

    overall_t1 = time.perf_counter()
    latency = {}
    latency.update({f"retrieval1_{k}": v for k, v in latency_retr.items()})
    latency.update({f"extract_{k}": v for k, v in latency_ext.items()})
    latency.update({f"retrieval2_{k}": v for k, v in latency_retr2.items()})
    latency["overall_ms"] = (overall_t1 - overall_t0) * 1000.0

    # shape output
    memories_out = [MemoryOut.model_validate(m) for m in active2]
    return RetrieveResponse(active_memories=memories_out, memory_context=context2, latency_ms=latency)
