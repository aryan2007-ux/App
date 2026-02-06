EXTRACT_SYSTEM = """You extract long-term memory from a single user turn for an AI assistant.

Store only information that will likely be useful later across long conversations:
- preferences (language, tone, format)
- constraints (availability, call-time windows)
- long-term instructions ("always...", "never...")
- commitments/reminders (time-bound; should have TTL)
- stable profile facts (name, timezone)

Do NOT store trivial or ephemeral facts.

Return STRICT JSON only:
{
  "ops": [
    {
      "action": "create|update|invalidate|noop",
      "type": "preference|constraint|instruction|commitment|profile|entity",
      "key": "language|call_time_window|timezone|tone|format|name|reminder|...",
      "value": <json>,
      "value_text": "<short canonical text for embedding>",
      "confidence": 0.0-1.0,
      "salience": 0.0-1.0,
      "ttl_hours": 0,
      "source_quote": "<short exact quote from the user message>"
    }
  ]
}
"""

EXTRACT_USER_TEMPLATE = """Turn meta:
- user_id: {user_id}
- turn_id: {turn_id}

User message:
\"\"\"{message}\"\"\"

Existing active memories summary:
{existing_summary}

Output JSON now."""
