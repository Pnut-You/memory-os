"""Run with: python3 example.py"""

from memory import MemoryConfig, MemoryManager


def main() -> None:
    config = MemoryConfig.from_env()
    config.redis_allow_memory_fallback = True  # Demo only; production should require Redis.
    memory = MemoryManager.create(config, start_scheduler=False)
    try:
        print(memory.search("user-001", "程序员", device_id="dog-001"))
    finally:
        memory.close()


if __name__ == "__main__":
    main()
