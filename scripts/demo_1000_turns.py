import random
import time

import httpx


def main():
    user_id = "demo_user"
    conversation_id = "default"
    base_url = "http://localhost:8000"

    topics = [
        "I prefer concise answers.",
        "My timezone is IST.",
        "Please always format code in Markdown.",
        "I like Python examples.",
        "Never use emojis.",
        "Remind me tomorrow to call mom.",
        "I can only take calls after 6pm.",
        "I work on a Mac.",
    ]

    with httpx.Client(timeout=30.0) as client:
        # health check
        r = client.get(f"{base_url}/health")
        r.raise_for_status()
        print("health:", r.json())

        for turn_id in range(1, 1001):
            msg = random.choice(topics)
            payload = {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "turn_id": turn_id,
                "message": msg,
            }
            r = client.post(f"{base_url}/turn", json=payload)
            r.raise_for_status()
            if turn_id % 50 == 0:
                data = r.json()
                print(f"turn {turn_id}: memories={len(data['active_memories'])} ctx_lines={data['memory_context'].count('\\n')+1 if data['memory_context'] else 0}")
            time.sleep(0.05)

    print("done")


if __name__ == "__main__":
    main()
