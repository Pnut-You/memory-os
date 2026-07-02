"""Run with: python3 example.py"""

import uuid

from memory import MemoryConfig, MemoryManager


def main() -> None:
    config = MemoryConfig.from_env()
    config.redis_allow_memory_fallback = True  # Demo only; production should require Redis.
    memory = MemoryManager.create(config, start_scheduler=False)
    try:
        memory.add_conversation_turn(
            f"demo-{uuid.uuid4().hex}",
            "user-001",
            "dog-001",
            "我喜欢安静一点的路线。",
            "好的，我会优先选择安静的路线。",
        )
        memory.events.upsert_preference(
            "user-001",
            "navigation.noise_level",
            "navigation",
            {"type": "enum", "code": "quiet", "label_zh": "安静"},
            "偏好安静路线",
            [],
            confidence=0.95,
        )
        memory.rebuild_user_card("user-001")
        print(memory.get_conversation_context("user-001", "dog-001"))
        print(memory.search("user-001", "安静"))
    finally:
        memory.close()


if __name__ == "__main__":
    main()
