import json
import time

import httpx

from app.models import ExtractResponse
from app.prompts import EXTRACT_SYSTEM, EXTRACT_USER_TEMPLATE
from app.settings import settings


def _existing_summary(existing: list[dict]) -> str:
    if not existing:
        return "(none)"
    lines = []
    for m in existing:
        lines.append(f"- [{m.get('type')}] {m.get('key')}: {m.get('value_text')}")
    return "\n".join(lines)


def extract_ops(user_id: str, turn_id: int, message: str, existing_active: list[dict]) -> tuple[ExtractResponse, dict[str, float]]:
    t0 = time.perf_counter()

    payload = {
        "model": settings.ollama_model,
        "format": "json",
        "stream": False,
        "messages": [
            {"role": "system", "content": EXTRACT_SYSTEM},
            {
                "role": "user",
                "content": EXTRACT_USER_TEMPLATE.format(
                    user_id=user_id,
                    turn_id=turn_id,
                    message=message,
                    existing_summary=_existing_summary(existing_active),
                ),
            },
        ],
    }

    t1 = time.perf_counter()
    with httpx.Client(timeout=90.0) as client:
        r = client.post(f"{settings.ollama_base_url}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()
    t2 = time.perf_counter()

    content = data["message"]["content"]
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {"ops": []}

    resp = ExtractResponse.model_validate(parsed)

    latency = {
        "build_prompt_ms": (t1 - t0) * 1000.0,
        "ollama_ms": (t2 - t1) * 1000.0,
        "total_ms": (t2 - t0) * 1000.0,
    }
    return resp, latency
