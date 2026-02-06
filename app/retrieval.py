import time

from app.db import get_conn
from app.embeddings import embed_text
from app.settings import settings


def fetch_active_memories(user_id: str) -> list[dict]:
    with get_conn() as conn:
        cur = conn.execute(
            """
            SELECT memory_id::text, type, key, value, value_text,
                   confidence, salience, source_turn_id, source_quote,
                   last_used_turn_id, times_used
            FROM memories
            WHERE user_id = %s AND status = 'active'
            ORDER BY salience DESC, updated_at DESC
            """,
            (user_id,),
        )
        return list(cur.fetchall())


def retrieve_context(user_id: str, message: str) -> tuple[list[dict], str, dict[str, float]]:
    t0 = time.perf_counter()

    active = fetch_active_memories(user_id)

    q_emb = embed_text(message)

    t1 = time.perf_counter()
    with get_conn() as conn:
        cur = conn.execute(
            """
            SELECT memory_id::text, type, key, value, value_text,
                   confidence, salience, source_turn_id, source_quote,
                   last_used_turn_id, times_used
            FROM memories
            WHERE user_id = %s AND status = 'active' AND embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (user_id, q_emb, settings.vector_top_k),
        )
        top = list(cur.fetchall())
    t2 = time.perf_counter()

    # Build a compact context block for the LLM.
    lines = []
    for m in top[: settings.max_memory_items]:
        lines.append(f"- [{m['type']}] {m['key']}: {m['value_text']}")
    context = "\n".join(lines) if lines else ""

    latency = {
        "embed_ms": (t1 - t0) * 1000.0,
        "db_ms": (t2 - t1) * 1000.0,
        "total_ms": (t2 - t0) * 1000.0,
    }
    return active, context, latency
