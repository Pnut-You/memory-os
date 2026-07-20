from __future__ import annotations

import json
import sqlite3
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from memory import MemoryConfig, MemoryManager, ShortTermMemory
from memory.migrate import migrate_jsonl
from memory.preferences import ActionPreferenceExtractionResult, PreferenceExtractionResult, PreferenceExtractor
from memory.sqlite_event import LEGACY_USER_ID, SQLiteEventStore
from ui.app import AgentConversationTurnRequest, QueryRequest
from ui.llm import DebugChatLLM
from ui.router import MemoryDebugRouter
from evaluation.run_preference_eval import answer_contains_expected


class BrokenRedisClient:
    def pipeline(self):
        raise RuntimeError("redis down")


class BlockingSummarizer:
    def __init__(self) -> None:
        import threading

        self.started = threading.Event()
        self.release = threading.Event()

    def summarize(self, messages, previous_summary=""):
        self.started.set()
        self.release.wait(5)
        return previous_summary + "\n异步摘要" if previous_summary else "异步摘要"


class CaptureSummarizer:
    def __init__(self) -> None:
        self.calls = []

    def summarize(self, messages, previous_summary=""):
        self.calls.append({"messages": messages, "previous_summary": previous_summary})
        ids = ",".join(str(item["id"]) for item in messages)
        return f"{previous_summary}\nsummary:{ids}".strip()


class FixedSummarizer:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = []

    def summarize(self, messages, previous_summary=""):
        self.calls.append({"messages": messages, "previous_summary": previous_summary})
        return self.text


class FakeExtractor:
    def __init__(self, result: dict | None = None, action_result: dict | None = None) -> None:
        self.calls = []
        self.action_calls = []
        self.model = "fake-70b"
        self.prompt_version = "test-v1"
        self._result = result or {"schema_version": "1.0", "user_id": "user-001", "preferences": []}
        self._action_result = action_result or {"schema_version": "1.0", "user_id": "user-001", "memories": []}

    def status(self):
        return {"enabled": True, "configured": True, "model": self.model}

    def extract(self, user_id, events, existing_preferences=None):
        self.calls.append({"user_id": user_id, "events": events, "existing_preferences": existing_preferences or []})
        data = dict(self._result)
        data["user_id"] = user_id
        return PreferenceExtractionResult.model_validate(data)

    @property
    def configured(self):
        return True

    def extract_action_preferences(self, user_id, action_memory_context):
        self.action_calls.append({"user_id": user_id, "action_memory_context": action_memory_context})
        data = dict(self._action_result)
        data["user_id"] = user_id
        return ActionPreferenceExtractionResult.model_validate(data)


class FailingExtractor(FakeExtractor):
    def extract(self, user_id, events, existing_preferences=None):
        self.calls.append({"user_id": user_id, "events": events, "existing_preferences": existing_preferences or []})
        raise RuntimeError("extractor down")


class FakeLLM:
    model = "qwen3.7-plus"

    def __init__(self) -> None:
        self.calls = []

    def status(self):
        return {"configured": True, "model": self.model}

    def complete(self, query, short_term, rolling_summary, user_card, latest_action_sequence=None):
        self.calls.append(
            {
                "query": query,
                "short_term": short_term,
                "rolling_summary": rolling_summary,
                "user_card": user_card,
                "latest_action_sequence": latest_action_sequence,
            }
        )
        return "好的，我会优先选择安静的路线。", {"model": self.model}


class RouteLLM(FakeLLM):
    def __init__(self, reply, event_routes=None):
        super().__init__()
        self.reply = reply
        self.event_routes = event_routes or []

    def complete(self, query, short_term, rolling_summary, user_card, latest_action_sequence=None):
        self.calls.append(
            {
                "query": query,
                "short_term": short_term,
                "rolling_summary": rolling_summary,
                "user_card": user_card,
                "latest_action_sequence": latest_action_sequence,
            }
        )
        return self.reply, {"model": self.model, "event_routes": self.event_routes}


class RecallNameLLM(FakeLLM):
    def complete(self, query, short_term, rolling_summary, user_card, latest_action_sequence=None):
        self.calls.append(
            {
                "query": query,
                "short_term": short_term,
                "rolling_summary": rolling_summary,
                "user_card": user_card,
                "latest_action_sequence": latest_action_sequence,
            }
        )
        for message in short_term:
            content = str(message.get("content") or "")
            marker = "机器狗叫"
            if marker in content:
                name = content.split(marker, 1)[1].split("。", 1)[0].split("，", 1)[0].strip()
                return f"你刚才说机器狗叫{name}。", {"model": self.model}
        return "你刚才没有提到机器狗的名字。", {"model": self.model}


class MemorySystemTests(unittest.TestCase):
    def test_example_does_not_write_demo_memory(self):
        source = (Path(__file__).parents[1] / "example.py").read_text(encoding="utf-8")
        self.assertNotIn("add_conversation_turn", source)
        self.assertNotIn("upsert_preference", source)
        self.assertNotIn("安静一点的路线", source)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.config = MemoryConfig(
            data_dir=root,
            sqlite_path=root / "events.db",
            local_long_term_path=root / "long_term.jsonl",
            redis_url="redis://127.0.0.1:1/0",
            redis_allow_memory_fallback=True,
            summary_every_turns=10,
            device_state_ttl_seconds=1,
            device_heartbeat_seconds=300,
            preference_extractor_enabled=False,
        )

    def tearDown(self):
        self.temp.cleanup()

    def make_manager(self) -> MemoryManager:
        return MemoryManager.create(self.config, start_scheduler=False)

    def add_short_memory_probe_turns(
        self,
        manager: MemoryManager,
        *,
        turn_count: int,
        fact_turn: int,
        fact_text: str,
    ) -> None:
        session_id = None
        for turn in range(1, turn_count + 1):
            if turn == fact_turn:
                user_text = fact_text
            else:
                user_text = f"这是第{turn}轮普通短会话。"
            result = manager.add_conversation_turn(
                f"short-probe-{turn}",
                "user-001",
                "dog-001",
                user_text,
                f"第{turn}轮回复。",
                session_id=session_id,
            )
            session_id = result["session_id"]

    def test_query_contract_requires_user_id_and_device_id(self):
        payload = QueryRequest(user_id="user-1", device_id="dog-1", query="你好")
        self.assertEqual(payload.user_id, "user-1")
        with self.assertRaises(Exception):
            QueryRequest(device_id="dog-1", query="你好")
        with self.assertRaises(Exception):
            QueryRequest(user_id="bad id", device_id="dog-1", query="你好")

    def test_agent_turn_contract_and_idempotent_write(self):
        payload = AgentConversationTurnRequest(
            request_id="agent-r1",
            user_id="user-001",
            device_id="dog-001",
            user_text="我喜欢飞盘",
            assistant_text="我记住了",
        )
        manager = self.make_manager()
        try:
            values = {
                "request_id": payload.request_id,
                "user_id": payload.user_id,
                "device_id": payload.device_id,
                "user_text": payload.user_text,
                "assistant_text": payload.assistant_text,
                "prompt_token_count": payload.prompt_token_count,
            }
            first = manager.add_agent_conversation_turn(**values)
            replay = manager.add_agent_conversation_turn(**values)
            self.assertFalse(first["idempotent_replay"])
            self.assertTrue(replay["idempotent_replay"])
            self.assertEqual(first["user_event_id"], replay["user_event_id"])
            self.assertEqual(len(manager.events.get_message_pair("agent-r1")), 2)
            with self.assertRaises(ValueError):
                manager.add_agent_conversation_turn(
                    "agent-r1", "user-001", "dog-001", "不同内容", "我记住了"
                )
        finally:
            manager.close()

    def test_agent_memory_context_returns_only_matching_dog_preferences(self):
        manager = self.make_manager()
        router = MemoryDebugRouter(manager, FakeLLM())
        try:
            manager.add_agent_conversation_turn(
                "agent-r1", "user-001", "dog-001", "当前狗的短期消息", "收到"
            )
            for dog, value in (("dog-001", "飞盘"), ("dog-002", "苹果")):
                manager.events.upsert_preference(
                    "user-001",
                    "preference.likes",
                    "preference",
                    {"type": "string", "value": value},
                    value,
                    [],
                    confidence=0.95,
                    device_id=dog,
                )
            manager.events.upsert_preference(
                "user-001",
                "profile.occupation",
                "profile",
                {"type": "string", "value": "程序员"},
                "程序员",
                [],
                confidence=0.95,
                device_id="dog-001",
            )
            manager.events.upsert_preference(
                "user-001",
                "preference.dislikes",
                "preference",
                {"type": "string", "value": "香菜"},
                "不喜欢吃香菜",
                [],
                confidence=0.95,
                device_id="dog-001",
            )
            context = router.agent_memory_context("user-001", "dog-001")
            self.assertEqual(
                context,
                {
                    "user_id": "user-001",
                    "device_id": "dog-001",
                    "long_term_memory": (
                        "个人信息：\n用户的职业是程序员。\n\n"
                        "偏好：\n用户的偏好包括：飞盘。\n\n"
                        "不喜欢：\n用户不喜欢：香菜。"
                    ),
                    "short_term_memory": {
                        "session_id": context["short_term_memory"]["session_id"],
                        "messages": [
                            {"role": "user", "content": "当前狗的短期消息"},
                            {"role": "assistant", "content": "收到"},
                        ],
                    },
                },
            )
        finally:
            manager.close()

    def test_agent_memory_prompt_keeps_empty_three_layer_structure(self):
        manager = self.make_manager()
        router = MemoryDebugRouter(manager, FakeLLM())
        try:
            context = router.agent_memory_context("empty-user", "dog-001")
            self.assertEqual(
                context["long_term_memory"],
                "个人信息：\n暂无已确认信息。\n\n"
                "偏好：\n暂无已确认信息。\n\n"
                "不喜欢：\n暂无已确认信息。",
            )
        finally:
            manager.close()

    def test_legacy_preferences_migrate_to_selected_dog_idempotently(self):
        manager = self.make_manager()
        try:
            manager.events.upsert_preference(
                "user-001",
                "preference.likes",
                "preference",
                {"type": "string", "value": "飞盘"},
                "飞盘",
                [],
                confidence=0.95,
            )
            self.assertEqual(manager.events.migrate_legacy_preferences("user-001", "dog-001"), 1)
            self.assertEqual(manager.events.migrate_legacy_preferences("user-001", "dog-001"), 0)
            self.assertEqual(len(manager.events.list_long_term_facts(
                "user-001", "preference.likes", device_id="dog-001"
            )), 1)
        finally:
            manager.close()

    def test_same_user_card_is_scoped_by_device(self):
        manager = self.make_manager()
        try:
            manager.events.upsert_preference(
                "user-001",
                "navigation.noise_level",
                "navigation",
                {"type": "enum", "code": "quiet", "label_zh": "安静"},
                "偏好安静路线",
                [],
                confidence=0.95,
                device_id="dog-1",
            )
            manager.rebuild_user_card("user-001", "dog-1")
            dog1 = manager.get_conversation_context("user-001", "dog-1")
            dog2 = manager.get_conversation_context("user-001", "dog-2")
            self.assertEqual(dog1["user_card"]["preferences"][0]["key"], "navigation.noise_level")
            self.assertIsNone(dog2["user_card"])
        finally:
            manager.close()

    def test_same_device_different_users_are_isolated(self):
        manager = self.make_manager()
        try:
            manager.add_conversation_turn("r1", "u1", "dog-1", "我喜欢安静", "记住")
            manager.add_conversation_turn("r2", "u2", "dog-1", "我喜欢热闹", "记住")
            c1 = manager.get_conversation_context("u1", "dog-1")
            c2 = manager.get_conversation_context("u2", "dog-1")
            self.assertIn("安静", c1["recent_messages"][0]["content"])
            self.assertIn("热闹", c2["recent_messages"][0]["content"])
            self.assertNotEqual(c1["recent_messages"][0]["content"], c2["recent_messages"][0]["content"])
        finally:
            manager.close()

    def test_conversation_cache_scoped_by_user_device_and_session(self):
        manager = self.make_manager()
        try:
            first = manager.add_conversation_turn(
                "r1",
                "u1",
                "dog-1",
                "a",
                "b",
                timestamp="2026-07-02T09:00:00+08:00",
            )
            second = manager.add_conversation_turn(
                "r2",
                "u1",
                "dog-1",
                "c",
                "d",
                timestamp="2026-07-02T09:00:16+08:00",
            )
            self.assertNotEqual(first["session_id"], second["session_id"])
            self.assertEqual(manager.redis.get_session_conversation("u1", "dog-1", first["session_id"])[0]["content"], "a")
            self.assertEqual(manager.redis.get_session_conversation("u1", "dog-1", second["session_id"])[0]["content"], "c")
        finally:
            manager.close()

    def test_session_timeout_closes_old_session_and_queues_it_once(self):
        manager = self.make_manager()
        try:
            first = manager.add_conversation_turn(
                "r1", "u1", "dog-1", "没有关键词也要检查", "好",
                timestamp="2026-07-02T09:00:00+08:00",
            )
            second = manager.add_conversation_turn(
                "r2", "u1", "dog-1", "新会话", "好",
                timestamp="2026-07-02T09:00:16+08:00",
                session_id=first["session_id"],
            )
            self.assertNotEqual(first["session_id"], second["session_id"])
            self.assertEqual(manager.events.get_session(first["session_id"])["status"], "closed")
            jobs = manager.events.list_jobs(job_type="preference_extraction")
            self.assertEqual(len(jobs), 1)
            self.assertEqual(jobs[0]["session_id"], first["session_id"])
            manager.resolve_session(
                "u1", "dog-1", session_id=first["session_id"],
                timestamp="2026-07-02T09:00:17+08:00",
            )
            self.assertEqual(len(manager.events.list_jobs(job_type="preference_extraction")), 1)
        finally:
            manager.close()

    def test_short_term_memory_does_not_cross_devices(self):
        manager = self.make_manager()
        try:
            first = manager.add_conversation_turn("dog1-r1", "user-001", "dog-001", "dog-001 的短期消息", "收到")
            second = manager.add_conversation_turn("dog2-r1", "user-001", "dog-002", "dog-002 的短期消息", "收到")
            dog1_cached = manager.redis.get_session_conversation("user-001", "dog-001", first["session_id"])
            dog2_cached = manager.redis.get_session_conversation("user-001", "dog-002", second["session_id"])
            self.assertTrue(any("dog-001" in item["content"] for item in dog1_cached))
            self.assertTrue(any("dog-002" in item["content"] for item in dog2_cached))
            self.assertEqual(manager.redis.get_session_conversation("user-001", "dog-002", first["session_id"]), [])
        finally:
            manager.close()

    def test_primary_long_term_preferences_are_scoped_by_device(self):
        manager = self.make_manager()
        try:
            manager.events.upsert_preference(
                "user-001",
                "preference.likes",
                "preference",
                {"type": "string", "code": "frisbee", "label_zh": "飞盘"},
                "喜欢飞盘",
                [],
                confidence=0.95,
                device_id="dog-001",
            )
            manager.rebuild_user_card("user-001", "dog-001")
            self.assertEqual(len(manager.events.list_long_term_facts(
                "user-001", "preference.likes", device_id="dog-001"
            )), 1)
            self.assertEqual(manager.events.list_preferences("user-001", status="active", device_id="dog-002"), [])
            manager.rebuild_user_card("user-001", "dog-002")
            self.assertIsNotNone(manager.get_user_card("user-001", "dog-001"))
            self.assertIsNone(manager.get_user_card("user-001", "dog-002"))
        finally:
            manager.close()

    def test_internal_long_term_write_revoke_and_card_restore(self):
        manager = self.make_manager()
        try:
            turn = manager.add_conversation_turn("r1", "user-001", "dog-001", "我喜欢飞盘", "好")
            pref_id = manager.write_long_term_preference(
                "user-001",
                "preference.likes",
                "飞盘",
                event_id=turn["user_event_id"],
                session_id=turn["session_id"],
            )
            self.assertIsNotNone(pref_id)
            card = manager.get_user_card("user-001", "dog-001")
            self.assertIn("preference.likes", {item["key"] for item in card["preferences"]})
            manager.revoke_long_term_preference(
                "user-001", "preference.likes", "飞盘", device_id="dog-001"
            )
            self.assertEqual(manager.events.list_long_term_facts(
                "user-001", "preference.likes", device_id="dog-001"
            ), [])
        finally:
            manager.close()

    def test_debug_user_long_term_memory_filters_by_device(self):
        manager = self.make_manager()
        router = MemoryDebugRouter(manager, FakeLLM())
        try:
            manager.events.upsert_preference(
                "user-001",
                "preference.likes",
                "preference",
                {"type": "string", "code": "frisbee", "label_zh": "飞盘"},
                "喜欢飞盘",
                [],
                confidence=0.95,
                device_id="dog-001",
            )
            manager.events.upsert_preference(
                "user-001",
                "preference.likes",
                "preference",
                {"type": "string", "code": "apple", "label_zh": "苹果"},
                "喜欢苹果",
                [],
                confidence=0.95,
                device_id="dog-002",
            )
            manager.rebuild_user_card("user-001", "dog-001")
            manager.rebuild_user_card("user-001", "dog-002")

            dog1 = router.debug_user("user-001", "dog-001")
            dog2 = router.debug_user("user-001", "dog-002")

            self.assertEqual(
                {item["display_text_zh"] for item in dog1["active_preferences"]},
                {"喜欢飞盘"},
            )
            self.assertEqual(
                {item["display_text_zh"] for item in dog2["active_preferences"]},
                {"喜欢苹果"},
            )
            self.assertEqual(dog1["user_card"]["device_id"], "dog-001")
            self.assertEqual(dog2["user_card"]["device_id"], "dog-002")
        finally:
            manager.close()

    def test_anonymous_does_not_create_preference_job(self):
        manager = self.make_manager()
        try:
            manager.add_conversation_turn("r1", "anonymous", "dog-1", "记住我喜欢安静", "好")
            self.assertEqual(manager.events.job_counts().get("pending", 0), 0)
        finally:
            manager.close()

    def test_message_count_does_not_create_preference_job_before_session_ends(self):
        manager = self.make_manager()
        try:
            for i in range(9):
                manager.add_conversation_turn(f"r{i}", "user-001", "dog-001", f"普通消息{i}", "好")
            self.assertEqual(manager.events.job_counts().get("pending", 0), 0)
            manager.add_conversation_turn("r9", "user-001", "dog-001", "普通消息9", "好")
            self.assertEqual(manager.events.job_counts().get("pending", 0), 0)
        finally:
            manager.close()

    def test_preference_keywords_do_not_schedule_before_session_ends(self):
        for text in ("我喜欢摄影", "我是摄影师", "我不喜欢吵闹"):
            manager = self.make_manager()
            try:
                manager.add_conversation_turn(f"r-{text}", "user-001", "dog-001", text, "好")
                self.assertEqual(manager.events.job_counts().get("pending", 0), 0, text)
            finally:
                manager.close()

    def test_memory_registry_contains_required_types(self):
        from memory.preferences import PREFERENCE_REGISTRY

        for key in (
            "profile.occupation",
            "preference.likes",
            "preference.dislikes",
            "habit.routine",
            "constraint.stable",
            "relationship.person",
            "default_behavior.preference",
        ):
            self.assertIn(key, PREFERENCE_REGISTRY)

    def test_query_does_not_call_extractor(self):
        manager = self.make_manager()
        extractor = FakeExtractor()
        manager.preference_extractor = extractor
        llm = FakeLLM()
        router = MemoryDebugRouter(manager, llm)
        try:
            result = router.submit("user-001", "dog-1", "记住我喜欢安静", debug=True)
            self.assertEqual(result["assistant_reply"], "好的，我会优先选择安静的路线。")
            self.assertEqual(extractor.calls, [])
            self.assertEqual(manager.events.job_counts().get("pending", 0), 0)
        finally:
            manager.close()

    def test_conversation_persistence_recovers_stale_sqlite_transaction(self):
        manager = self.make_manager()
        try:
            with manager.events._lock:
                manager.events._conn.execute("BEGIN")
                self.assertTrue(manager.events._conn.in_transaction)
            result = manager.add_conversation_turn("r-stale", "user-001", "dog-001", "你好", "你好")
            self.assertGreater(result["user_event_id"], 0)
            self.assertFalse(manager.events._conn.in_transaction)
            events = manager.events.list_events(user_id="user-001", device_id="dog-001", event_type="message", limit=10)
            self.assertEqual(len(events), 2)
        finally:
            manager.close()

    def test_preference_extraction_runs_in_background(self):
        manager = self.make_manager()
        manager.preference_extractor = FakeExtractor(
            {
                "schema_version": "1.0",
                "user_id": "user-001",
                "preferences": [
                    {
                        "preference_key": "navigation.noise_level",
                        "category": "navigation",
                        "value": {"type": "enum", "code": "quiet", "label_zh": "安静"},
                        "display_text_zh": "偏好安静路线",
                        "confidence": 0.96,
                        "strength": 0.85,
                        "evidence": [{"event_id": 1, "text": "我喜欢安静路线", "type": "explicit"}],
                    }
                ],
            }
        )
        try:
            manager.add_conversation_turn(
                "r1", "user-001", "dog-1", "记住我喜欢安静路线", "好",
                timestamp="2026-07-02T09:00:00+08:00",
            )
            manager.get_conversation_context(
                "user-001", "dog-1", timestamp="2026-07-02T09:00:16+08:00"
            )
            self.assertEqual(manager.process_memory_jobs_once()["succeeded"], 1)
            self.assertEqual(manager.process_memory_jobs_once()["succeeded"], 1)
            prefs = manager.events.list_preferences("user-001", status="active", device_id="dog-1")
            self.assertEqual(prefs[0]["preference_key"], "navigation.noise_level")
            self.assertIsNotNone(manager.get_user_card("user-001", "dog-1"))
        finally:
            manager.close()

    def test_action_habit_text_does_not_schedule_without_preference_keyword(self):
        manager = self.make_manager()
        manager.preference_extractor = FakeExtractor()
        try:
            result = manager.add_conversation_turn(
                "r-action-pref",
                "user-001",
                "dog-001",
                "以后默认往前走然后坐下",
                "好",
            )
            self.assertNotIn("action_event_id", result)
            self.assertEqual(manager.events.list_jobs(job_type="preference_extraction"), [])
            self.assertEqual(manager.events.list_action_events("user-001", "dog-001"), [])
        finally:
            manager.close()

    def test_manual_preference_extraction_creates_and_runs_user_job(self):
        manager = self.make_manager()
        manager.preference_extractor = FakeExtractor(
            {
                "schema_version": "1.0",
                "user_id": "user-001",
                "preferences": [
                    {
                        "preference_key": "interaction.reply_length",
                        "category": "interaction",
                        "value": {"type": "enum", "code": "short", "label_zh": "简短"},
                        "display_text_zh": "喜欢简短回复",
                        "confidence": 0.95,
                        "strength": 0.8,
                        "reason_zh": "用户连续表达喜欢简短回答",
                        "evidence": [{"event_id": 1, "text": "我喜欢你回答简短一点", "type": "explicit"}],
                    }
                ],
            }
        )
        router = MemoryDebugRouter(manager, FakeLLM())
        try:
            router.submit("user-001", "dog-001", "我喜欢你回答简短一点", debug=True)
            result = router.extract_user_preferences("user-001", "dog-001")
            self.assertEqual(result["process"]["succeeded"], 1)
            prefs = result["memory"]["active_preferences"]
            self.assertEqual(prefs[0]["preference_key"], "interaction.reply_length")
            self.assertIn("喜欢简短回复", prefs[0]["display_text_zh"])
        finally:
            manager.close()

    def test_three_structured_long_term_memory_keys_are_stored(self):
        manager = self.make_manager()
        manager.preference_extractor = FakeExtractor(
            {
                "schema_version": "1.0",
                "user_id": "user-001",
                "preferences": [
                    {
                        "preference_key": "profile.occupation",
                        "category": "profile",
                        "value": {"type": "string", "code": "photographer", "label_zh": "摄影师"},
                        "display_text_zh": "职业是摄影师",
                        "confidence": 0.95,
                        "strength": 0.8,
                        "evidence": [{"event_id": 1, "text": "我是摄影师", "type": "explicit"}],
                    },
                    {
                        "preference_key": "preference.likes",
                        "category": "preference",
                        "value": {"type": "string", "code": "photography", "label_zh": "摄影"},
                        "display_text_zh": "喜欢摄影",
                        "confidence": 0.95,
                        "strength": 0.8,
                        "evidence": [{"event_id": 1, "text": "我喜欢摄影", "type": "explicit"}],
                    },
                    {
                        "preference_key": "preference.dislikes",
                        "category": "preference",
                        "value": {"type": "string", "code": "noise", "label_zh": "吵闹"},
                        "display_text_zh": "不喜欢吵闹",
                        "confidence": 0.95,
                        "strength": 0.8,
                        "evidence": [{"event_id": 1, "text": "我不喜欢吵闹", "type": "explicit"}],
                    },
                ],
            }
        )
        router = MemoryDebugRouter(manager, FakeLLM())
        try:
            router.submit("user-001", "dog-001", "我喜欢摄影", debug=True)
            result = router.extract_user_preferences("user-001", "dog-001")
            keys = {item["preference_key"] for item in result["memory"]["active_preferences"]}
            self.assertIn("profile.occupation", keys)
            self.assertIn("preference.likes", keys)
            self.assertIn("preference.dislikes", keys)
            card_keys = [item["key"] for item in manager.get_user_card("user-001", "dog-001")["preferences"]]
            self.assertIn("profile.occupation", card_keys)
        finally:
            manager.close()

    def test_manual_preference_extraction_force_reprocesses_recent_window(self):
        manager = self.make_manager()
        manager.preference_extractor = FakeExtractor()
        router = MemoryDebugRouter(manager, FakeLLM())
        try:
            router.submit("user-001", "dog-001", "我喜欢你回答简短一点", debug=True)
            first = router.extract_user_preferences("user-001", "dog-001")
            self.assertEqual(first["process"]["succeeded"], 1)
            self.assertEqual(first["latest_global_event_id"], first["to_event_id"])

            second = router.extract_user_preferences("user-001", "dog-001")
            self.assertEqual(second["mode"], "force_recent")
            self.assertEqual(second["process"]["succeeded"], 1)
            self.assertEqual(len(manager.preference_extractor.calls), 2)
            context = manager.preference_extractor.calls[-1]["events"]
            texts = [
                message["text"]
                for turn in context["recent_turns"]
                for message in turn["messages"]
            ]
            self.assertIn("我喜欢你回答简短一点", texts)
        finally:
            manager.close()

    def test_closed_session_is_not_reprocessed_without_new_messages(self):
        manager = self.make_manager()
        manager.preference_extractor = FakeExtractor()
        try:
            manager.add_conversation_turn(
                "r1", "user-001", "dog-001", "记住我喜欢安静", "好",
                timestamp="2026-07-02T09:00:00+08:00",
            )
            manager.get_conversation_context(
                "user-001", "dog-001", timestamp="2026-07-02T09:00:16+08:00"
            )
            self.assertEqual(manager.process_memory_jobs_once()["succeeded"], 1)
            result = manager.trigger_preference_extraction("user-001", "dog-001", force_recent=False)
            self.assertEqual(result["message"], "没有可抽取的新用户消息")
            self.assertEqual(result["process"]["claimed"], 0)
        finally:
            manager.close()

    def test_manual_preference_extraction_keeps_debug_context(self):
        manager = self.make_manager()
        manager.summarizer = CaptureSummarizer()
        manager.preference_extractor = FakeExtractor()
        try:
            for i in range(20):
                text = "我喜欢安静路线" if i == 0 else f"普通消息{i}"
                manager.add_conversation_turn(
                    f"r{i}",
                    "user-001",
                    "dog-001",
                    text,
                    "好",
                    prompt_token_count=6000,
                )
            self.assertTrue(manager.wait_for_summaries())
            result = manager.trigger_preference_extraction("user-001", "dog-001", force_recent=True)
            self.assertEqual(result["process"]["succeeded"], 1)
            context = manager.preference_extractor.calls[-1]["events"]
            self.assertEqual(context["context_mode"], "closed_session_user_messages")
            self.assertIn("summary:", context["rolling_summary"])
            texts = [
                message["text"]
                for turn in context["recent_turns"]
                for message in turn["messages"]
            ]
            self.assertIn("普通消息19", texts)
        finally:
            manager.close()

    def test_summary_window_dislike_becomes_structured_preference(self):
        manager = self.make_manager()
        manager.summarizer = FixedSummarizer("用户不喜欢吃香菜，有独特的饮食偏好。")
        manager.preference_extractor = FakeExtractor(
            {
                "schema_version": "1.0",
                "user_id": "user-001",
                "preferences": [
                    {
                        "preference_key": "preference.dislikes",
                        "category": "preference",
                        "value": {"type": "string", "code": "香菜", "label_zh": "香菜"},
                        "display_text_zh": "不喜欢香菜",
                        "confidence": 0.95,
                        "strength": 0.8,
                        "polarity": "avoid",
                        "evidence": [{"event_id": 1, "text": "我不喜欢吃香菜", "type": "explicit"}],
                    }
                ],
            }
        )
        try:
            manager.add_conversation_turn("r0", "user-001", "dog-001", "我不喜欢吃香菜", "记住了")
            for i in range(1, 10):
                manager.add_conversation_turn(f"r{i}", "user-001", "dog-001", f"普通消息{i}", "好")
            self.assertTrue(manager.wait_for_summaries())
            manager.get_conversation_context(
                "user-001", "dog-001", timestamp="2099-01-01T00:00:00+08:00"
            )
            result = manager.process_memory_jobs_once()
            self.assertGreaterEqual(result["succeeded"], 1)
            prefs = manager.events.list_preferences("user-001", status="active")
            dislike = [item for item in prefs if item["preference_key"] == "preference.dislikes"]
            self.assertTrue(dislike)
            self.assertIn("香菜", dislike[0]["display_text_zh"])
        finally:
            manager.close()

    def test_stale_running_preference_job_is_recovered(self):
        manager = self.make_manager()
        manager.preference_extractor = FakeExtractor()
        try:
            manager.add_conversation_turn(
                "r1", "user-001", "dog-001", "记住我喜欢安静", "好",
                timestamp="2026-07-02T09:00:00+08:00",
            )
            manager.get_conversation_context(
                "user-001", "dog-001", timestamp="2026-07-02T09:00:16+08:00"
            )
            claimed = manager.events.claim_jobs(1, max_attempts=3)
            self.assertEqual(claimed[0]["job_type"], "preference_extraction")
            with manager.events._lock:
                manager.events._conn.execute(
                    "UPDATE memory_jobs SET locked_at=? WHERE id=?",
                    ("2000-01-01T00:00:00+00:00", claimed[0]["id"]),
                )
                manager.events._conn.commit()
            result = manager.process_memory_jobs_once()
            self.assertEqual(result["recovered_stale"], 1)
            self.assertEqual(result["succeeded"], 1)
        finally:
            manager.close()

    def test_failed_preference_job_stops_after_max_attempts(self):
        manager = self.make_manager()
        manager.preference_extractor = FailingExtractor()
        try:
            manager.add_conversation_turn(
                "r1", "user-001", "dog-001", "记住这件事", "好",
                timestamp="2026-07-02T09:00:00+08:00",
            )
            manager.get_conversation_context(
                "user-001", "dog-001", timestamp="2026-07-02T09:00:16+08:00"
            )
            for _ in range(manager.preference_extract_max_attempts):
                result = manager.process_memory_jobs_once()
                self.assertEqual(result["failed"], 1)
            exhausted = manager.process_memory_jobs_once()
            self.assertEqual(exhausted["claimed"], 0)
        finally:
            manager.close()

    def test_preference_job_fails_when_model_fails_without_local_rules(self):
        manager = self.make_manager()
        manager.preference_extractor = FailingExtractor()
        try:
            manager.add_conversation_turn(
                "r1", "user-001", "dog-001", "我喜欢吃苹果", "好",
                timestamp="2026-07-02T09:00:00+08:00",
            )
            manager.get_conversation_context(
                "user-001", "dog-001", timestamp="2026-07-02T09:00:16+08:00"
            )
            result = manager.process_memory_jobs_once()
            self.assertEqual(result["succeeded"], 0)
            self.assertEqual(result["failed"], 1)
            self.assertIn("extractor down", result["errors"][0]["error"])
            prefs = manager.events.list_preferences("user-001", status="active")
            self.assertEqual(prefs, [])
        finally:
            manager.close()

    def test_manual_preference_extraction_returns_structured_error_when_job_create_fails(self):
        manager = self.make_manager()
        try:
            manager.add_conversation_turn("r1", "user-001", "dog-001", "我喜欢安静", "好")
            original = manager.events.restart_preference_extraction_job

            def fail(*args, **kwargs):
                raise RuntimeError("database is locked")

            manager.events.restart_preference_extraction_job = fail
            result = manager.trigger_preference_extraction("user-001", "dog-001")
            self.assertEqual(result["process"]["failed"], 1)
            self.assertIn("database is locked", result["process"]["errors"][0]["error"])
            manager.events.restart_preference_extraction_job = original
        finally:
            manager.close()

    def test_preference_result_accepts_array_or_single_item_without_user_id(self):
        array_result = PreferenceExtractionResult.model_validate_json(
            json.dumps(
                [
                    {
                        "preference_key": "interaction.reply_length",
                        "category": "interaction",
                        "value": {"type": "enum", "code": "short"},
                        "display_text_zh": "喜欢简短回复",
                        "confidence": 0.9,
                        "strength": 0.8,
                    }
                ],
                ensure_ascii=False,
            ),
            default_user_id="u",
        )
        self.assertEqual(array_result.user_id, "u")
        self.assertEqual(array_result.preferences[0].preference_key, "interaction.reply_length")
        single_result = PreferenceExtractionResult.model_validate_json(
            json.dumps(
                {
                    "preference_key": "interaction.style",
                    "category": "interaction",
                    "value": {"type": "enum", "code": "direct"},
                    "display_text_zh": "喜欢直接回复",
                    "confidence": 0.9,
                    "strength": 0.8,
                },
                ensure_ascii=False,
            ),
            default_user_id="u",
        )
        self.assertEqual(single_result.user_id, "u")
        self.assertEqual(single_result.preferences[0].preference_key, "interaction.style")

    def test_config_prefers_dashscope_key_and_preference_reuses_it(self):
        root = Path(self.temp.name)
        env_path = root / ".env"
        env_path.write_text(
            "\n".join(
                [
                    "DASHSCOPE_API_KEY=sk-dashscope-key",
                    "LLM_API_KEY=sk-llm-key",
                    "LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "LLM_MODEL=qwen3.7-plus",
                ]
            ),
            encoding="utf-8",
        )
        with patch.dict("os.environ", {}, clear=True):
            config = MemoryConfig.from_env(env_path)
        self.assertEqual(config.llm_api_key, "sk-dashscope-key")
        self.assertEqual(config.llm_api_key_source, "DASHSCOPE_API_KEY")
        self.assertTrue(config.preference_extractor_enabled)
        self.assertEqual(config.preference_extractor_api_key, "sk-dashscope-key")
        self.assertEqual(config.preference_extractor_api_key_source, "DASHSCOPE_API_KEY")
        self.assertEqual(config.preference_extractor_base_url, "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.assertEqual(config.preference_extractor_model, "qwen3.5-flash-2026-02-23")
        self.assertEqual(config.long_term_small_model, "qwen3.5-flash-2026-02-23")
        self.assertEqual(config.long_term_large_model, "qwen3.5-flash-2026-02-23")

    def test_config_keeps_llm_api_key_as_compatibility_fallback(self):
        root = Path(self.temp.name)
        env_path = root / ".env"
        env_path.write_text("LLM_API_KEY=sk-legacy-key\n", encoding="utf-8")
        with patch.dict("os.environ", {}, clear=True):
            config = MemoryConfig.from_env(env_path)
        self.assertEqual(config.llm_api_key, "sk-legacy-key")
        self.assertEqual(config.llm_api_key_source, "LLM_API_KEY")
        self.assertEqual(config.llm_model, "qwen3.5-flash-2026-02-23")
        self.assertEqual(config.preference_extractor_api_key, "sk-legacy-key")
        self.assertEqual(config.preference_extractor_api_key_source, "LLM_API_KEY")

    def test_debug_llm_status_and_invalid_dashscope_key_are_diagnostic(self):
        llm = DebugChatLLM(
            "2783-invalid",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "qwen3.7-plus",
            "LLM_API_KEY",
        )
        status = llm.status()
        self.assertEqual(status["api_key_source"], "LLM_API_KEY")
        self.assertEqual(status["api_key_hint"]["prefix"], "2783")
        self.assertEqual(status["api_key_hint"]["length"], len("2783-invalid"))
        with self.assertRaisesRegex(RuntimeError, "LLM_API_KEY"):
            llm.complete("你好", [], "", {})

    def test_debug_llm_does_not_change_shared_prompt_for_long_term_verification(self):
        llm = DebugChatLLM("", "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen3.7-plus")
        empty_messages = llm.build_messages("用户的职业是什么？", [], "", {})
        empty_prompt = "\n".join(item["content"] for item in empty_messages)
        self.assertNotIn("长期记忆事实", empty_prompt)

        messages = llm.build_messages(
            "用户的职业是什么？",
            [{"role": "user", "content": "当前会话事实"}],
            "",
            {
                "preferences": [
                    {"key": "profile.occupation", "value": "programmer", "label_zh": "职业"},
                    {"key": "preference.likes", "value": "羽毛球", "label_zh": "喜欢羽毛球"},
                    {"key": "preference.dislikes", "value": "香菜", "label_zh": "不喜欢香菜"},
                ]
            },
        )
        prompt = "\n".join(item["content"] for item in messages)
        self.assertNotIn("长期记忆事实", prompt)
        self.assertIn("当前会话事实", prompt)

    def test_preference_verification_match_allows_minor_function_words(self):
        self.assertTrue(answer_contains_expected("我记得您不喜欢吵闹的环境。", "吵闹环境"))
        self.assertTrue(answer_contains_expected("根据我的记忆，您不喜欢刺耳的提示音。", "刺耳提示音"))
        self.assertFalse(answer_contains_expected("你的职业是医生。", "程序员"))

    def test_session_preference_output_rejects_cross_event_evidence(self):
        raw = json.dumps({"preferences": [{
            "event_id": 1,
            "type": "occupation",
            "value": "程序员",
            "evidence": "我的职业是程序员",
            "confidence": 0.99,
        }]}, ensure_ascii=False)
        with self.assertRaisesRegex(ValueError, "referenced user text"):
            PreferenceExtractor._validate_session_output(
                raw,
                [{"event_id": 1, "text": "我的职业是医生"}, {"event_id": 2, "text": "我的职业是程序员"}],
            )

    def test_hybrid_preference_extractor_falls_back_after_validation_failure(self):
        extractor = PreferenceExtractor(
            enabled=True,
            api_key="test",
            base_url="https://example.test/v1",
            model="legacy-large",
            mode="hybrid",
            small_model="small-model",
            large_model="large-model",
        )
        invalid = json.dumps({"preferences": [{
            "event_id": 1, "type": "likes", "value": "摄影",
            "evidence": "我喜欢摄影", "confidence": 0.9,
        }]}, ensure_ascii=False)
        valid = json.dumps({"preferences": [{
            "event_id": 1, "type": "likes", "value": "羽毛球",
            "evidence": "我喜欢羽毛球", "confidence": 0.95,
        }]}, ensure_ascii=False)
        with patch.object(
            extractor, "_request_session_extraction",
            side_effect=[(invalid, {}), (valid, {})],
        ) as request:
            result = extractor.extract(
                "u",
                {"source_user_events": [{"event_id": 1, "text": "我喜欢羽毛球"}]},
            )
        self.assertEqual(request.call_args_list[0].args[0], "small-model")
        self.assertEqual(request.call_args_list[1].args[0], "large-model")
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.preferences[0].value["value"], "羽毛球")

    def test_session_extractor_sends_all_user_messages_in_one_request(self):
        extractor = PreferenceExtractor(
            enabled=True,
            api_key="test",
            base_url="https://example.test/v1",
            model="flash",
        )
        raw = json.dumps({"preferences": [{
            "event_id": 2,
            "type": "likes",
            "value": "安静",
            "evidence": "这种环境挺安静，我很享受",
            "confidence": 0.92,
        }]}, ensure_ascii=False)
        with patch.object(
            extractor, "_request_session_extraction", return_value=(raw, {"prompt_tokens": 42})
        ) as request:
            result = extractor.extract("u", {"source_user_events": [
                {"event_id": 1, "text": "今天聊点别的"},
                {"event_id": 2, "text": "这种环境挺安静，我很享受"},
            ]})
        self.assertEqual(request.call_count, 1)
        self.assertEqual(request.call_args.args[1][0]["event_id"], 1)
        self.assertEqual(len(request.call_args.args[1]), 2)
        self.assertEqual(result.preferences[0].value["value"], "安静")
        self.assertEqual(result.request_metrics[0]["usage"]["prompt_tokens"], 42)

    def test_valid_none_does_not_trigger_hybrid_fallback(self):
        extractor = PreferenceExtractor(
            enabled=True,
            api_key="test",
            base_url="https://example.test/v1",
            model="legacy-large",
            mode="hybrid",
        )
        raw = json.dumps({"preferences": []})
        with patch.object(extractor, "_request_session_extraction", return_value=(raw, {})) as request:
            result = extractor.extract(
                "u",
                {"source_user_events": [{"event_id": 1, "text": "我是来聊天的"}]},
            )
        self.assertEqual(request.call_count, 1)
        self.assertFalse(result.fallback_used)
        self.assertEqual(result.preferences, [])

    def test_long_term_fact_query_returns_sqlite_values_without_calling_llm(self):
        manager = self.make_manager()
        llm = FakeLLM()
        router = MemoryDebugRouter(manager, llm)
        try:
            for value in ("羽毛球", "电子音乐"):
                manager.events.upsert_preference(
                    "u",
                    "preference.likes",
                    "preference",
                    {"type": "string", "value": value, "code": value, "label_zh": value},
                    value,
                    [],
                    confidence=0.95,
                    device_id="dog-1",
                )
            result = router.submit("u", "dog-1", "根据长期记忆，你记得我喜欢什么？", debug=True)
            self.assertEqual(set(result["assistant_reply"].split("、")), {"羽毛球", "电子音乐"})
            self.assertEqual(llm.calls, [])
            self.assertEqual(result["model"], "sqlite-long-term-facts")
            self.assertEqual(result["debug"]["retrieved_long_term_facts"]["field"], "likes")
            self.assertEqual(manager.events.list_jobs(job_type="preference_extraction"), [])
        finally:
            manager.close()

    def test_short_term_fact_question_is_not_intercepted(self):
        manager = self.make_manager()
        llm = FakeLLM()
        router = MemoryDebugRouter(manager, llm)
        try:
            result = router.submit("u", "dog-1", "我刚才说我喜欢什么？")
            self.assertEqual(result["assistant_reply"], "好的，我会优先选择安静的路线。")
            self.assertEqual(len(llm.calls), 1)
        finally:
            manager.close()

    def test_required_long_term_values_do_not_cross_users_or_fields(self):
        manager = self.make_manager()
        try:
            facts = [
                ("previous-user", "profile.occupation", "程序员"),
                ("current-user", "profile.occupation", "医生"),
                ("current-user", "preference.likes", "羽毛球"),
                ("current-user", "preference.likes", "电子音乐"),
                ("current-user", "preference.dislikes", "香菜"),
            ]
            for user_id, key, value in facts:
                manager.events.upsert_preference(
                    user_id,
                    key,
                    "profile" if key == "profile.occupation" else "preference",
                    {"type": "string", "value": value, "code": value, "label_zh": value},
                    value,
                    [],
                    confidence=0.95,
                    device_id="dog-1",
                )
            self.assertEqual(manager.answer_long_term_fact("current-user", "dog-1", "occupation")[0], "医生")
            self.assertEqual(
                set(manager.answer_long_term_fact("current-user", "dog-1", "likes")[0].split("、")),
                {"羽毛球", "电子音乐"},
            )
            self.assertEqual(manager.answer_long_term_fact("current-user", "dog-1", "dislikes")[0], "香菜")
            self.assertEqual(manager.answer_long_term_fact("missing-user", "dog-1", "likes")[0], "未记录")
        finally:
            manager.close()

    def test_delete_user_memory_removes_all_device_user_cards(self):
        manager = self.make_manager()
        try:
            manager.redis.set_json(manager.redis.user_card_key("u", "dog-1"), {"value": 1}, 60)
            manager.redis.set_json(manager.redis.user_card_key("u", "dog-2"), {"value": 2}, 60)
            manager.delete_user_memory("u")
            self.assertIsNone(manager.get_user_card("u", "dog-1"))
            self.assertIsNone(manager.get_user_card("u", "dog-2"))
        finally:
            manager.close()

    def test_same_preference_increments_evidence_count(self):
        manager = self.make_manager()
        try:
            e1, _ = manager.events.add_message_pair("r1", "u", "d", "我喜欢安静", "好")
            e2, _ = manager.events.add_message_pair("r2", "u", "d", "还是安静", "好")
            for event_id in (e1, e2):
                manager.events.upsert_preference(
                    "u",
                    "navigation.noise_level",
                    "navigation",
                    {"type": "enum", "code": "quiet"},
                    "偏好安静路线",
                    [{"event_id": event_id, "text": "安静", "type": "explicit"}],
                    confidence=0.95,
                )
            pref = manager.events.list_preferences("u", status="active")[0]
            self.assertEqual(pref["evidence_count"], 2)
        finally:
            manager.close()

    def test_conflicting_preference_supersedes_old(self):
        manager = self.make_manager()
        try:
            manager.events.upsert_preference("u", "navigation.noise_level", "navigation", {"type": "enum", "code": "quiet"}, "偏好安静路线", [], confidence=0.95)
            manager.events.upsert_preference("u", "navigation.noise_level", "navigation", {"type": "enum", "code": "busy"}, "偏好热闹路线", [], confidence=0.95)
            active = manager.events.list_preferences("u", status="active")[0]
            superseded = manager.events.list_preferences("u", status="superseded")[0]
            self.assertEqual(active["value_json"]["code"], "busy")
            self.assertEqual(active["supersedes_id"], superseded["id"])
        finally:
            manager.close()

    def test_likes_and_dislikes_keep_multiple_active_values(self):
        manager = self.make_manager()
        try:
            manager.events.upsert_preference(
                "u",
                "preference.likes",
                "preference",
                {"type": "string", "code": "travel", "label_zh": "旅游"},
                "喜欢旅游",
                [],
                confidence=0.95,
            )
            manager.events.upsert_preference(
                "u",
                "preference.likes",
                "preference",
                {"type": "string", "code": "apple", "label_zh": "苹果"},
                "喜欢吃苹果",
                [],
                confidence=0.95,
            )
            manager.events.upsert_preference(
                "u",
                "preference.dislikes",
                "preference",
                {"type": "string", "code": "cilantro", "label_zh": "香菜"},
                "不喜欢香菜",
                [],
                polarity="avoid",
                confidence=0.95,
            )
            active = manager.events.list_preferences("u", status="active", limit=10)
            labels = {item["display_text_zh"] for item in active}
            self.assertIn("喜欢旅游", labels)
            self.assertIn("喜欢吃苹果", labels)
            self.assertIn("不喜欢香菜", labels)
        finally:
            manager.close()

    def test_model_extraction_does_not_get_local_rule_supplements(self):
        manager = self.make_manager()
        manager.preference_extractor = FakeExtractor(
            {
                "schema_version": "1.0",
                "user_id": "user-001",
                "preferences": [
                    {
                        "preference_key": "preference.likes",
                        "category": "preference",
                        "value": {"type": "string", "code": "travel", "label_zh": "旅游"},
                        "display_text_zh": "喜欢旅游",
                        "confidence": 0.95,
                        "strength": 0.8,
                        "evidence": [{"event_id": 1, "text": "我喜欢旅游", "type": "explicit"}],
                    }
                ],
            }
        )
        router = MemoryDebugRouter(manager, FakeLLM())
        try:
            router.submit("user-001", "dog-001", "我喜欢吃苹果", debug=True)
            result = router.extract_user_preferences("user-001", "dog-001")
            self.assertEqual(result["process"]["succeeded"], 1)
            active = manager.events.list_preferences("user-001", status="active", limit=10)
            labels = {item["display_text_zh"] for item in active}
            self.assertIn("喜欢旅游", labels)
            self.assertNotIn("喜欢苹果", labels)
        finally:
            manager.close()

    def test_same_like_display_with_different_value_json_does_not_duplicate(self):
        manager = self.make_manager()
        try:
            manager.events.upsert_preference(
                "u",
                "preference.likes",
                "preference",
                {"type": "string", "code": "travel", "label_zh": "旅游"},
                "喜欢旅游",
                [],
                confidence=0.95,
            )
            manager.events.upsert_preference(
                "u",
                "preference.likes",
                "preference",
                {"type": "string", "code": "旅游"},
                "喜欢旅游",
                [],
                confidence=0.95,
            )
            active = manager.events.list_preferences("u", status="active", limit=10)
            self.assertEqual([item["display_text_zh"] for item in active], ["喜欢旅游"])
            self.assertEqual(active[0]["evidence_count"], 2)
        finally:
            manager.close()

    def test_repeated_multi_value_preference_increments_evidence_count(self):
        manager = self.make_manager()
        try:
            e1, _ = manager.events.add_message_pair("r-like-1", "u", "d", "我喜欢吃苹果", "好")
            e2, _ = manager.events.add_message_pair("r-like-2", "u", "d", "我还是喜欢吃苹果", "好")
            apple = {"type": "string", "code": "apple", "label_zh": "苹果"}
            for event_id in (e1, e2):
                manager.events.upsert_preference(
                    "u",
                    "preference.likes",
                    "preference",
                    apple,
                    "喜欢吃苹果",
                    [{"event_id": event_id, "text": "我喜欢吃苹果", "type": "explicit"}],
                    confidence=0.95,
                )
            active = manager.events.list_preferences("u", status="active")
            self.assertEqual(len(active), 1)
            self.assertEqual(active[0]["evidence_count"], 2)
        finally:
            manager.close()

    def test_occupation_is_single_active_value(self):
        manager = self.make_manager()
        try:
            manager.events.upsert_preference(
                "u",
                "profile.occupation",
                "profile",
                {"type": "string", "code": "programmer", "label_zh": "程序员"},
                "职业是程序员",
                [],
                confidence=0.95,
            )
            manager.events.upsert_preference(
                "u",
                "profile.occupation",
                "profile",
                {"type": "string", "code": "doctor", "label_zh": "医生"},
                "职业是医生",
                [],
                confidence=0.95,
            )
            active = manager.events.list_preferences("u", status="active")
            superseded = manager.events.list_preferences("u", status="superseded")
            self.assertEqual(len(active), 1)
            self.assertEqual(active[0]["value_json"]["code"], "doctor")
            self.assertEqual(superseded[0]["value_json"]["code"], "programmer")
            self.assertEqual(active[0]["supersedes_id"], superseded[0]["id"])
        finally:
            manager.close()

    def test_dislike_revokes_matching_like_without_removing_other_likes(self):
        manager = self.make_manager()
        try:
            manager.events.upsert_preference(
                "u",
                "preference.likes",
                "preference",
                {"type": "string", "code": "travel", "label_zh": "旅游"},
                "喜欢旅游",
                [],
                confidence=0.95,
            )
            manager.events.upsert_preference(
                "u",
                "preference.likes",
                "preference",
                {"type": "string", "code": "apple", "label_zh": "苹果"},
                "喜欢吃苹果",
                [],
                confidence=0.95,
            )
            manager.events.upsert_preference(
                "u",
                "preference.dislikes",
                "preference",
                {"type": "string", "code": "apple", "label_zh": "苹果"},
                "不喜欢吃苹果",
                [],
                polarity="avoid",
                confidence=0.95,
            )
            active = manager.events.list_preferences("u", status="active", limit=10)
            revoked = manager.events.list_preferences("u", status="revoked", limit=10)
            active_labels = {item["display_text_zh"] for item in active}
            revoked_labels = {item["display_text_zh"] for item in revoked}
            self.assertIn("喜欢旅游", active_labels)
            self.assertIn("不喜欢吃苹果", active_labels)
            self.assertNotIn("喜欢吃苹果", active_labels)
            self.assertIn("喜欢吃苹果", revoked_labels)
        finally:
            manager.close()

    def test_revoke_multi_value_preference_is_exact(self):
        manager = self.make_manager()
        try:
            travel = {"type": "string", "code": "travel", "label_zh": "旅游"}
            apple = {"type": "string", "code": "apple", "label_zh": "苹果"}
            manager.events.upsert_preference("u", "preference.likes", "preference", travel, "喜欢旅游", [], confidence=0.95)
            manager.events.upsert_preference("u", "preference.likes", "preference", apple, "喜欢吃苹果", [], confidence=0.95)
            manager.events.revoke_preference("u", "preference.likes", apple)
            active_labels = {
                item["display_text_zh"]
                for item in manager.events.list_preferences("u", status="active", limit=10)
            }
            revoked_labels = {
                item["display_text_zh"]
                for item in manager.events.list_preferences("u", status="revoked", limit=10)
            }
            self.assertEqual(active_labels, {"喜欢旅游"})
            self.assertEqual(revoked_labels, {"喜欢吃苹果"})
        finally:
            manager.close()

    def test_temporary_preference_has_expires_at(self):
        manager = self.make_manager()
        try:
            expires = "2026-06-26T00:00:00+00:00"
            manager.events.upsert_preference(
                "u",
                "interaction.reply_length",
                "interaction",
                {"type": "enum", "code": "short"},
                "临时简短回复",
                [],
                durability="temporary",
                expires_at=expires,
                confidence=0.9,
            )
            pref = manager.events.list_preferences("u", status="active")[0]
            self.assertEqual(pref["expires_at"], expires)
        finally:
            manager.close()

    def test_redis_loss_restores_user_card_and_recent_context(self):
        manager = self.make_manager()
        try:
            manager.events.upsert_preference(
                "u",
                "interaction.language",
                "interaction",
                {"type": "enum", "code": "zh"},
                "默认中文",
                [],
                confidence=0.95,
                device_id="d",
            )
            manager.rebuild_user_card("u", "d")
            for i in range(3):
                manager.add_conversation_turn(f"r{i}", "u", "d", f"问题{i}", f"回答{i}")
            manager.redis.delete_key(manager.redis.user_card_key("u", "d"))
            manager.redis.clear_conversation("d", "u")
            ctx = manager.get_conversation_context("u", "d")
            self.assertEqual(ctx["user_card"]["preferences"][0]["key"], "interaction.language")
            self.assertEqual(len(ctx["recent_messages"]), 6)
        finally:
            manager.close()

    def test_rolling_summary_does_not_block_main_request(self):
        manager = self.make_manager()
        blocker = BlockingSummarizer()
        manager.summarizer = blocker
        try:
            for i in range(19):
                manager.add_conversation_turn(f"r{i}", "u", "d", f"问题{i}", f"回答{i}", prompt_token_count=6000)
            started = time.perf_counter()
            manager.add_conversation_turn("r19", "u", "d", "问题19", "回答19", prompt_token_count=6000)
            self.assertLess(time.perf_counter() - started, 0.5)
            self.assertTrue(blocker.started.wait(1))
        finally:
            blocker.release.set()
            manager.close()

    def test_short_summary_skips_when_twenty_turns_but_prompt_under_threshold(self):
        manager = self.make_manager()
        capture = CaptureSummarizer()
        manager.summarizer = capture
        try:
            for i in range(20):
                manager.add_conversation_turn(f"r{i}", "u", "d", f"问题{i}", f"回答{i}", prompt_token_count=4999)
            self.assertTrue(manager.wait_for_summaries())
            self.assertIsNone(manager.events.latest_summary("u", "d"))
            self.assertEqual(capture.calls, [])
            events = manager.events.list_events(user_id="u", device_id="d", event_type="message", limit=100)
            self.assertTrue(any(item["content"] == "问题0" for item in events))
        finally:
            manager.close()

    def test_short_summary_skips_when_under_twenty_turns_even_if_prompt_large(self):
        manager = self.make_manager()
        capture = CaptureSummarizer()
        manager.summarizer = capture
        try:
            for i in range(19):
                manager.add_conversation_turn(f"r{i}", "u", "d", f"问题{i}", f"回答{i}", prompt_token_count=6000)
            self.assertTrue(manager.wait_for_summaries())
            self.assertIsNone(manager.events.latest_summary("u", "d"))
            self.assertEqual(capture.calls, [])
        finally:
            manager.close()

    def test_ten_turn_session_enters_prompt_without_truncating_first_fact(self):
        manager = self.make_manager()
        llm = RecallNameLLM()
        router = MemoryDebugRouter(manager, llm)
        try:
            self.add_short_memory_probe_turns(
                manager,
                turn_count=10,
                fact_turn=1,
                fact_text="我的机器狗叫小黑。",
            )
            result = router.submit("user-001", "dog-001", "机器狗叫什么？", debug=True)
            prompt_text = "\n".join(item["content"] for item in result["debug"]["prompt_messages"])
            self.assertIn("小黑", prompt_text)
            self.assertIn("小黑", result["assistant_reply"])
            self.assertEqual(len(llm.calls[0]["short_term"]), 20)
            self.assertIn("小黑", llm.calls[0]["short_term"][0]["content"])
        finally:
            manager.close()

    def test_fourteen_turn_session_enters_prompt_without_truncating_middle_fact(self):
        manager = self.make_manager()
        llm = RecallNameLLM()
        router = MemoryDebugRouter(manager, llm)
        try:
            self.add_short_memory_probe_turns(
                manager,
                turn_count=14,
                fact_turn=4,
                fact_text="我的机器狗叫可乐。",
            )
            result = router.submit("user-001", "dog-001", "机器狗叫什么？", debug=True)
            prompt_text = "\n".join(item["content"] for item in result["debug"]["prompt_messages"])
            self.assertIn("可乐", prompt_text)
            self.assertIn("可乐", result["assistant_reply"])
            self.assertEqual(len(llm.calls[0]["short_term"]), 28)
            self.assertTrue(any("可乐" in item["content"] for item in llm.calls[0]["short_term"]))
        finally:
            manager.close()

    def test_twenty_turn_session_under_token_trigger_enters_prompt_without_summary_or_truncation(self):
        manager = self.make_manager()
        capture = CaptureSummarizer()
        manager.summarizer = capture
        llm = RecallNameLLM()
        router = MemoryDebugRouter(manager, llm)
        try:
            self.add_short_memory_probe_turns(
                manager,
                turn_count=20,
                fact_turn=1,
                fact_text="我的机器狗叫小黑。",
            )
            result = router.submit("user-001", "dog-001", "机器狗叫什么？", debug=True)
            self.assertLess(result["debug"]["prompt_token_count"], 5000)
            self.assertIsNone(manager.events.latest_summary("user-001", "dog-001"))
            self.assertEqual(capture.calls, [])
            self.assertEqual(len(llm.calls[0]["short_term"]), 40)
            prompt_text = "\n".join(item["content"] for item in result["debug"]["prompt_messages"])
            for turn in range(1, 21):
                expected = "我的机器狗叫小黑。" if turn == 1 else f"这是第{turn}轮普通短会话。"
                self.assertIn(expected, prompt_text)
        finally:
            manager.close()

    def test_short_summary_compacts_old_conversation_and_keeps_recent_five_turns(self):
        manager = self.make_manager()
        capture = CaptureSummarizer()
        manager.summarizer = capture
        try:
            for i in range(20):
                manager.add_conversation_turn(f"r{i}", "u", "d", f"问题{i}", f"回答{i}", prompt_token_count=6000)
            self.assertTrue(manager.wait_for_summaries())
            first = manager.events.latest_summary("u", "d")
            self.assertEqual(first["version"], 1)
            self.assertEqual(first["from_event_id"], 1)
            self.assertEqual(first["to_event_id"], 30)
            self.assertEqual(first["compacted_through_event_id"], 30)
            self.assertEqual(first["turn_count"], 15)
            self.assertEqual([item["id"] for item in capture.calls[0]["messages"]], list(range(1, 31)))
            ctx = manager.get_conversation_context("u", "d")
            self.assertEqual(ctx["recent_messages"][0]["id"], 31)
            self.assertEqual(ctx["recent_messages"][-1]["id"], 40)
            self.assertTrue(all(f"问题{i}" in [item["content"] for item in ctx["recent_messages"]] for i in range(15, 20)))
        finally:
            manager.close()

    def test_summary_does_not_keep_compacting_without_twenty_new_turns(self):
        manager = self.make_manager()
        capture = CaptureSummarizer()
        manager.summarizer = capture
        try:
            for i in range(30):
                manager.add_conversation_turn(f"r{i}", "u", "d", f"问题{i}", f"回答{i}", prompt_token_count=6000)
            manager.wait_for_summaries()
            self.assertEqual(len(capture.calls), 1)
            last_ids = [item["id"] for item in capture.calls[-1]["messages"]]
            self.assertEqual(last_ids[0], 1)
            self.assertLessEqual(last_ids[-1], 50)
            self.assertTrue(all(item["role"] in {"user", "assistant"} for item in capture.calls[-1]["messages"]))
        finally:
            manager.close()

    def test_short_summary_is_scoped_by_local_date(self):
        manager = self.make_manager()
        capture = CaptureSummarizer()
        manager.summarizer = capture
        try:
            for i in range(20):
                manager.add_conversation_turn(
                    f"day1-{i}",
                    "u",
                    "d",
                    f"第一天问题{i}",
                    f"第一天回答{i}",
                    timestamp=f"2026-07-02T09:00:{i:02d}+08:00",
                    prompt_token_count=6000,
                )
            self.assertTrue(manager.wait_for_summaries())
            first = manager.events.latest_summary("u", "d", "2026-07-02")
            self.assertIsNotNone(first)
            self.assertEqual(first["local_date"], "2026-07-02")
            self.assertEqual(first["version"], 1)

            for i in range(20):
                manager.add_conversation_turn(
                    f"day2-{i}",
                    "u",
                    "d",
                    f"第二天问题{i}",
                    f"第二天回答{i}",
                    timestamp=f"2026-07-03T09:00:{i:02d}+08:00",
                    prompt_token_count=6000,
                )
            self.assertTrue(manager.wait_for_summaries())
            second = manager.events.latest_summary("u", "d", "2026-07-03")
            self.assertIsNotNone(second)
            self.assertEqual(second["local_date"], "2026-07-03")
            self.assertEqual(second["version"], 1)
            self.assertNotEqual(first["id"], second["id"])
            self.assertTrue(all("第二天" in item["content"] for item in capture.calls[-1]["messages"]))
            ctx = manager.get_conversation_context("u", "d", timestamp="2026-07-03T09:00:20+08:00")
            self.assertEqual(ctx["summary_version"], 1)
            self.assertEqual(ctx["rolling_summary"], second["summary_text"])
            self.assertNotEqual(ctx["rolling_summary"], first["summary_text"])
        finally:
            manager.close()

    def test_local_summary_has_hard_length_limit(self):
        manager = self.make_manager()
        try:
            for i in range(30):
                manager.add_conversation_turn(
                    f"long-{i}",
                    "u",
                    "d",
                    f"问题{i}" + "很长的内容" * 80,
                    f"回答{i}" + "很长的回复" * 80,
                    prompt_token_count=6000,
                )
                manager.wait_for_summaries()
            latest = manager.events.latest_summary("u", "d")
            self.assertLessEqual(len(latest["summary_text"]), 1600)
            self.assertIn("日期总结", latest["summary_text"])
        finally:
            manager.close()

    def test_session_redis_list_keeps_latest_ten_turns_and_sqlite_full_text(self):
        manager = self.make_manager()
        try:
            for i in range(12):
                manager.add_conversation_turn(f"r{i}", "u", "d", f"完整问题{i}", f"完整回答{i}")
            session_id = manager.events.list_sessions("u", "d")[0]["session_id"]
            cached = manager.redis.get_session_conversation("u", "d", session_id)
            self.assertEqual(len(cached), 24)
            self.assertEqual(cached[0]["content"], "完整问题0")
            events = manager.events.list_events(user_id="u", device_id="d", event_type="message", limit=100)
            self.assertEqual(len(events), 24)
            self.assertTrue(any(e["content"] == "完整问题0" for e in events))
        finally:
            manager.close()

    def test_tool_chain_and_device_state_are_preserved(self):
        manager = self.make_manager()
        try:
            run_id = manager.begin_tool_run("dog-1:u1", "navigate", {"target": "东门"}, idempotency_key="nav-1")
            self.assertEqual(manager.begin_tool_run("dog-1:u1", "navigate", {}, idempotency_key="nav-1"), run_id)
            manager.record_tool_step(run_id, "planned", {"distance": 120})
            manager.finish_tool_run(run_id, {"arrived": True})
            durable = manager.get_tool_run(run_id)
            self.assertEqual(durable["status"], "completed")
            first = datetime.now(timezone.utc)
            initial = manager.update_device_state("dog-1", {"battery": 80}, first.isoformat())
            heartbeat = manager.update_device_state("dog-1", {"battery": 80}, (first + timedelta(seconds=301)).isoformat())
            self.assertEqual(initial["reason"], "initial")
            self.assertEqual(heartbeat["reason"], "heartbeat")
            self.assertEqual(len(manager.get_device_history("dog-1")), 2)
            time.sleep(1.05)
            self.assertFalse(manager.get_device_state("dog-1")["online"])
        finally:
            manager.close()

    def test_device_realtime_state_core_fields(self):
        manager = self.make_manager()
        router = MemoryDebugRouter(manager, FakeLLM())
        try:
            state = {
                "battery_percent": 76,
                "charging": False,
                "network": "wifi",
                "location": "客厅",
                "motion_state": "idle",
                "temperature_c": 36.5,
            }
            result = router.update_debug_device_state("dog-001", state, "2026-06-25T10:00:00+08:00")
            self.assertTrue(result["state"]["online"])
            self.assertEqual(result["state"]["state"]["battery_percent"], 76)
            unchanged = router.update_debug_device_state("dog-001", state, "2026-06-25T10:00:10+08:00")
            self.assertFalse(unchanged["updated"]["history_written"])
            changed = dict(state, battery_percent=75)
            changed_result = router.update_debug_device_state("dog-001", changed, "2026-06-25T10:00:20+08:00")
            self.assertEqual(changed_result["updated"]["reason"], "change")
            self.assertEqual(len(manager.get_device_history("dog-001")), 2)
        finally:
            manager.close()

    def test_ui_contains_device_realtime_state_page(self):
        html = Path("ui/static/index.html").read_text(encoding="utf-8")
        self.assertIn("设备实时状态", html)
        self.assertIn("长期记忆", html)
        self.assertIn("短期记忆", html)
        self.assertIn("结构化偏好", html)
        self.assertIn("职业", html)
        self.assertIn("明确不喜欢", html)
        self.assertIn("groupedPrefTable", html)
        self.assertIn("请求链路", html)
        self.assertIn("trace-view", html)
        self.assertIn("偏好抽取上下文", html)
        self.assertIn("run-preference-extract", html)
        self.assertIn("device_id（用于设备隔离）", html)
        self.assertIn('dev=$("memory-device").value.trim()', html)
        self.assertIn("?device_id=${encodeURIComponent(dev)}", html)
        self.assertIn("/api/debug/users/${encodeURIComponent(u)}${q}", html)
        self.assertIn("错误:", html)
        self.assertIn("模型警告:", html)
        self.assertIn("日期总结", html)
        self.assertIn("事件记忆库", html)
        self.assertIn("日期事件记忆", html)
        self.assertIn("7天事件偏好记忆", html)
        self.assertIn("run-time-extract", html)
        self.assertIn("run-action-extract", html)
        self.assertIn("run-weekly-action-preferences", html)
        self.assertIn("/api/debug/users/${encodeURIComponent(user_id)}/events/extract", html)
        self.assertIn("action_memory", html)
        self.assertIn("action_preference_memory", html)
        self.assertNotIn("Session 动作记忆", html)
        self.assertNotIn("session-action-memories", html)
        self.assertIn("selectedSessionId", html)
        self.assertIn("data-session-id", html)
        self.assertIn("/api/memories/events-text", html)
        self.assertNotIn('id="events-type"', html)
        self.assertNotIn('<option value="time_memory"', html)
        self.assertNotIn('<option value="event_summary"', html)
        self.assertNotIn('<option value="message"', html)
        self.assertNotIn('<option value="action_chain_summary"', html)
        self.assertNotIn("payload_json?.actions", html)
        self.assertNotIn("动作记忆事件库", html)
        self.assertNotIn("摘要正文", html)
        self.assertNotIn("add-time-memory", html)
        self.assertNotIn("add-event-summary", html)
        self.assertNotIn("手动新增", html)
        self.assertNotIn("目标时间", html)
        self.assertNotIn("时间任务列表", html)
        self.assertNotIn("定时任务", html)
        self.assertNotIn("条件任务", html)
        self.assertNotIn("待补全事件", html)
        self.assertIn("await r.text()", html)
        self.assertIn('rel="icon"', html)
        self.assertIn("@media(max-width:1200px)", html)
        self.assertIn("@media(max-width:640px)", html)
        self.assertNotIn("overflow:hidden", html)
        for field in (
            "device-battery",
            "device-charging",
            "device-network",
            "device-location",
            "device-motion",
            "device-temperature",
        ):
            self.assertIn(field, html)

    def test_ui_favicon_route_does_not_404(self):
        try:
            from fastapi.testclient import TestClient
        except ModuleNotFoundError as exc:
            self.skipTest(f"fastapi test client unavailable: {exc}")

        from ui.app import app

        response = TestClient(app).get("/favicon.ico")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/svg+xml")
        self.assertIn("<svg", response.text)

    def test_time_memory_is_daily_text_summary(self):
        manager = self.make_manager()
        try:
            memory_at = "2026-07-02T21:00:00+08:00"
            event_id = manager.remember_at(
                "user-001",
                "dog-001",
                "今天完成了客厅巡检，并确认用户喜欢安静路线。",
                memory_at,
                {"memory_date": "2026-07-02", "title": "当天活动总结"},
            )
            memories = manager.events.list_time_memories("user-001", "dog-001")
            self.assertEqual(memories[0]["id"], event_id)
            self.assertEqual(memories[0]["event_type"], "time_memory")
            self.assertEqual(memories[0]["payload_json"]["memory_at"], memory_at)
            self.assertEqual(memories[0]["payload_json"]["memory_date"], "2026-07-02")
            self.assertEqual(memories[0]["payload_json"]["title"], "当天活动总结")
            self.assertEqual(memories[0]["content"], "今天完成了客厅巡检，并确认用户喜欢安静路线。")
        finally:
            manager.close()

    def test_plain_text_no_longer_creates_time_pending_or_action_event(self):
        manager = self.make_manager()
        try:
            result = manager.add_conversation_turn(
                "plain-route",
                "user-001",
                "dog-001",
                "明天早上九点提醒我喝水，然后坐下",
                "好的",
            )
            self.assertNotIn("pending_event", result)
            self.assertIsNone(manager.redis.get_value("pending-event", "dog-001:user-001"))
            self.assertEqual(manager.events.list_time_memories("user-001", "dog-001"), [])
            self.assertEqual(manager.events.list_action_events("user-001", "dog-001"), [])
        finally:
            manager.close()

    def test_old_scheduled_tasks_are_not_returned_as_time_memories(self):
        manager = self.make_manager()
        try:
            manager.events.add_event(
                "old-task",
                "user-001",
                "dog-001",
                "scheduled_task",
                {"target_at": "2026-07-02T09:00:00+08:00", "task": "喝水"},
                content="喝水",
            )
            self.assertEqual(manager.events.list_time_memories("user-001", "dog-001"), [])
        finally:
            manager.close()

    def test_model_scheduled_route_is_ignored_by_new_time_memory(self):
        manager = self.make_manager()
        try:
            result = manager.add_conversation_turn(
                "model-time-ignored",
                "user-001",
                "dog-001",
                "随便聊聊",
                "好的",
                timestamp="2026-07-01T10:00:00+08:00",
                model_event_routes=[
                    {
                        "type": "scheduled_task",
                        "decision": "create",
                        "confidence": 0.9,
                        "task": "喝水",
                        "target_at": "2026-07-02T09:00:00+08:00",
                    }
                ],
            )
            self.assertNotIn("pending_event", result)
            self.assertIsNone(manager.redis.get_value("pending-event", "dog-001:user-001"))
            self.assertEqual(manager.events.list_time_memories("user-001", "dog-001"), [])
        finally:
            manager.close()

    def test_event_summary_is_separate_text_event(self):
        manager = self.make_manager()
        try:
            event_id = manager.events.add_event_summary(
                "user-001",
                "dog-001",
                "完成一次外出巡检，用户确认路线偏好。",
                "2026-07-02T18:00:00+08:00",
                "外出巡检",
            )
            events = manager.events.list_event_summaries("user-001", "dog-001")
            self.assertEqual(events[0]["id"], event_id)
            self.assertEqual(events[0]["event_type"], "event_summary")
            self.assertEqual(events[0]["payload_json"]["event_at"], "2026-07-02T18:00:00+08:00")
            self.assertEqual(events[0]["payload_json"]["title"], "外出巡检")
        finally:
            manager.close()

    def test_model_event_route_candidate_failure_still_replies(self):
        manager = self.make_manager()
        llm = RouteLLM("好的", [{"type": "scheduled_task", "decision": "create", "confidence": 0.4, "missing_fields": []}])
        router = MemoryDebugRouter(manager, llm)
        try:
            result = router.submit("user-001", "dog-001", "随便聊聊", debug=True)
            self.assertEqual(result["assistant_reply"], "好的")
            action_step = next(step for step in result["debug"]["trace_steps"] if step["name"] == "action_event_routing")
            self.assertEqual(action_step["status"], "skipped")
        finally:
            manager.close()

    def test_model_event_route_candidate_only_records_actions(self):
        manager = self.make_manager()
        llm = RouteLLM(
            "好的",
            [
                {
                    "type": "action_sequence",
                    "decision": "create",
                    "confidence": 0.8,
                    "actions": [{"code": "sit", "label_zh": "坐下"}],
                    "missing_fields": [],
                }
            ],
        )
        router = MemoryDebugRouter(manager, llm)
        try:
            result = router.submit("user-001", "dog-001", "随便聊聊", debug=True)
            self.assertEqual(result["assistant_reply"], "好的")
            self.assertEqual(len(manager.events.list_action_events("user-001", "dog-001")), 1)
            self.assertIsNone(manager.redis.get_value("pending-event", "dog-001:user-001"))
        finally:
            manager.close()

    def test_query_debug_trace_includes_time_memory_flow(self):
        manager = self.make_manager()
        router = MemoryDebugRouter(
            manager,
            RouteLLM(
                "好的",
                [
                    {
                        "type": "scheduled_task",
                        "decision": "create",
                        "confidence": 0.9,
                        "task": "叫我起床",
                        "target_at": "2026-07-02T09:00:00+08:00",
                    }
                ],
            ),
        )
        try:
            result = router.submit("user-001", "dog-001", "明天早上9点钟要叫我起床", debug=True)
            steps = result["debug"]["trace_steps"]
            names = [step["name"] for step in steps]
            self.assertIn("request_input", names)
            self.assertIn("rolling_summary", names)
            self.assertIn("long_term_memory", names)
            self.assertIn("daily_memory_extraction", names)
            self.assertIn("llm_prompt_messages", names)
            time_step = next(step for step in steps if step["name"] == "daily_memory_extraction")
            self.assertEqual(time_step["status"], "deferred")
            self.assertEqual(time_step["title_zh"], "日期总结抽取")
            self.assertRegex(time_step["data"]["memory_date"], r"^\d{4}-\d{2}-\d{2}$")
        finally:
            manager.close()

    def test_plain_query_does_not_create_time_memory(self):
        manager = self.make_manager()
        try:
            manager.add_conversation_turn("r-plain", "user-001", "dog-001", "你好", "你好")
            self.assertEqual(manager.events.list_time_memories("user-001", "dog-001"), [])
        finally:
            manager.close()

    def test_daily_time_memory_is_extracted_from_day_session_history(self):
        manager = self.make_manager()
        manager.summarizer = FixedSummarizer("当天完成了客厅巡检，并确认用户喜欢安静路线。")
        try:
            result = manager.add_conversation_turn(
                "daily-1",
                "user-001",
                "dog-001",
                "今天先巡检客厅",
                "已完成客厅巡检",
                timestamp="2026-07-02T09:00:00+08:00",
            )
            self.assertIn("daily_extraction", result)
            self.assertTrue(result["daily_extraction"]["deferred"])
            self.assertEqual(manager.events.list_time_memories("user-001", "dog-001"), [])
            scheduled = manager.schedule_due_memory_jobs(
                datetime(2026, 7, 3, 1, 0, tzinfo=timezone(timedelta(hours=8)))
            )
            self.assertEqual(len(scheduled["daily_created"]), 2)
            process = manager.process_memory_jobs_once(limit=2, include_daily=True)
            self.assertEqual(process["succeeded"], 2)
            memories = manager.events.list_time_memories("user-001", "dog-001")
            self.assertEqual(len(memories), 1)
            self.assertEqual(memories[0]["payload_json"]["memory_date"], "2026-07-02")
            self.assertIn("当天完成了客厅巡检", memories[0]["content"])
            self.assertEqual(memories[0]["payload_json"]["metadata"]["message_count"], 2)
        finally:
            manager.close()

    def test_daily_scheduler_waits_until_one_and_is_idempotent(self):
        manager = self.make_manager()
        try:
            manager.add_conversation_turn(
                "scheduled-daily-boundary",
                "user-001",
                "dog-001",
                "巡检客厅",
                "已完成",
                timestamp="2026-07-02T20:00:00+08:00",
            )
            before = manager.schedule_due_memory_jobs(
                datetime(2026, 7, 3, 0, 59, tzinfo=timezone(timedelta(hours=8)))
            )
            self.assertEqual(before["backfill_end"], "2026-07-01")
            self.assertEqual(before["daily_created"], [])
            due = manager.schedule_due_memory_jobs(
                datetime(2026, 7, 3, 1, 0, tzinfo=timezone(timedelta(hours=8)))
            )
            self.assertEqual(
                {item["job_type"] for item in due["daily_created"]},
                {"daily_time_memory_extract", "daily_action_memory_extract"},
            )
            duplicate = manager.schedule_due_memory_jobs(
                datetime(2026, 7, 3, 8, 0, tzinfo=timezone(timedelta(hours=8)))
            )
            self.assertEqual(duplicate["daily_created"], [])
            jobs = manager.events.list_jobs("user-001", limit=20)
            self.assertEqual(len([job for job in jobs if job.get("schedule_key") == "2026-07-02"]), 2)
        finally:
            manager.close()

    def test_scheduler_backfills_only_latest_thirty_completed_days(self):
        manager = self.make_manager()
        try:
            manager.add_conversation_turn(
                "backfill-too-old",
                "user-001",
                "dog-001",
                "旧对话",
                "旧回复",
                timestamp="2026-06-01T09:00:00+08:00",
            )
            manager.add_conversation_turn(
                "backfill-in-range",
                "user-001",
                "dog-001",
                "近期对话",
                "近期回复",
                timestamp="2026-06-02T09:00:00+08:00",
            )
            result = manager.schedule_due_memory_jobs(
                datetime(2026, 7, 2, 1, 0, tzinfo=timezone(timedelta(hours=8)))
            )
            self.assertEqual(result["backfill_start"], "2026-06-02")
            keys = {item["memory_date"] for item in result["daily_created"]}
            self.assertEqual(keys, {"2026-06-02"})
        finally:
            manager.close()

    def test_daily_time_memory_rerun_replaces_same_day_summary(self):
        manager = self.make_manager()
        manager.summarizer = FixedSummarizer("第一版总结")
        try:
            manager.add_conversation_turn(
                "daily-replace-1",
                "user-001",
                "dog-001",
                "上午巡检",
                "完成",
                timestamp="2026-07-02T09:00:00+08:00",
            )
            manager.process_memory_jobs_once(limit=2, include_daily=True)
            manager.summarizer = FixedSummarizer("第二版总结")
            result = manager.trigger_daily_extraction("user-001", "dog-001", "2026-07-02")
            self.assertEqual(result["process"]["succeeded"], 1)
            memories = manager.events.list_time_memories("user-001", "dog-001")
            self.assertEqual(len(memories), 1)
            self.assertIn("第二版总结", memories[0]["content"])
        finally:
            manager.close()

    def test_daily_time_memory_local_fallback_is_not_full_transcript(self):
        manager = self.make_manager()
        try:
            manager.add_conversation_turn(
                "daily-local-summary",
                "user-001",
                "dog-001",
                "今天先巡检客厅，然后提醒我检查门窗",
                "这是助手逐句回复，应该不要作为时间记忆原样保存",
                timestamp="2026-07-02T09:00:00+08:00",
            )
            result = manager.trigger_daily_extraction("user-001", "dog-001", "2026-07-02")
            self.assertEqual(result["process"]["succeeded"], 1)
            memory = manager.events.list_time_memories("user-001", "dog-001")[0]
            self.assertIn("日期总结", memory["content"])
            self.assertIn("巡检客厅", memory["content"])
            self.assertNotIn("助手逐句回复", memory["content"])
            self.assertEqual(memory["payload_json"]["metadata"]["summary_backend"], "local")
        finally:
            manager.close()

    def test_daily_action_sequences_create_action_memories(self):
        manager = self.make_manager()
        try:
            for idx in range(2):
                manager.add_conversation_turn(
                    f"chain-{idx}",
                    "user-001",
                    "dog-001",
                    "往前走然后坐下",
                    "好的",
                    timestamp=f"2026-07-02T09:0{idx}:00+08:00",
                    model_event_routes=[
                        {
                            "type": "action_sequence",
                            "decision": "create",
                            "confidence": 0.9,
                            "actions": [
                                {"code": "forward", "label_zh": "往前走"},
                                {"code": "sit", "label_zh": "坐下"},
                            ],
                        }
                    ],
                )
            result = manager.trigger_daily_event_extraction("user-001", "dog-001", "2026-07-02")
            self.assertEqual(result["process"]["succeeded"], 1)
            memories = manager.events.list_action_memories("user-001", "dog-001", memory_date="2026-07-02")
            self.assertEqual(len(memories), 1)
            self.assertEqual(memories[0]["event_type"], "action_memory")
            self.assertEqual(memories[0]["payload_json"]["memory_date"], "2026-07-02")
            self.assertEqual(memories[0]["payload_json"]["metadata"]["action_chain_count"], 2)
            self.assertIn("1. 用户要求 往前走 -> 坐下", memories[0]["content"])
            self.assertIn("2. 用户要求 往前走 -> 坐下", memories[0]["content"])
        finally:
            manager.close()

    def test_daily_event_button_extracts_from_day_conversation_without_action_route(self):
        manager = self.make_manager()
        try:
            manager.add_conversation_turn(
                "daily-event-from-text",
                "user-001",
                "dog-001",
                "请你坐下，然后转圈",
                "好的，我坐下后转圈。",
                timestamp="2026-07-02T09:00:00+08:00",
            )
            self.assertEqual(manager.events.list_action_memories("user-001", "dog-001"), [])
            result = manager.trigger_daily_event_extraction("user-001", "dog-001", "2026-07-02")
            self.assertEqual(result["process"]["succeeded"], 1)
            self.assertEqual(result["event_memory_count"], 1)
            memory = manager.events.list_action_memories("user-001", "dog-001", memory_date="2026-07-02")[0]
            self.assertIn("请你坐下", memory["content"])
            self.assertEqual(memory["payload_json"]["metadata"]["extract_backend"], "local")
        finally:
            manager.close()

    def test_daily_action_memory_rerun_replaces_single_day_text(self):
        manager = self.make_manager()
        try:
            manager.add_conversation_turn(
                "daily-action-replace-1",
                "user-001",
                "dog-001",
                "坐下",
                "好的",
                timestamp="2026-07-02T09:00:00+08:00",
                model_event_routes=[
                    {
                        "type": "action_sequence",
                        "decision": "create",
                        "confidence": 0.9,
                        "actions": [{"code": "sit", "label_zh": "坐下"}],
                    }
                ],
            )
            manager.trigger_daily_event_extraction("user-001", "dog-001", "2026-07-02")
            first = manager.events.list_action_memories("user-001", "dog-001", memory_date="2026-07-02")
            self.assertEqual(len(first), 1)
            self.assertIn("坐下", first[0]["content"])

            manager.add_conversation_turn(
                "daily-action-replace-2",
                "user-001",
                "dog-001",
                "转圈",
                "好的",
                timestamp="2026-07-02T09:00:05+08:00",
                session_id=first[0]["payload_json"]["metadata"]["session_ids"][0],
                model_event_routes=[
                    {
                        "type": "action_sequence",
                        "decision": "create",
                        "confidence": 0.9,
                        "actions": [{"code": "spin", "label_zh": "转圈"}],
                    }
                ],
            )
            result = manager.trigger_daily_event_extraction("user-001", "dog-001", "2026-07-02")
            self.assertEqual(result["event_memory_count"], 1)
            replaced = manager.events.list_action_memories("user-001", "dog-001", memory_date="2026-07-02")
            self.assertEqual(len(replaced), 1)
            self.assertIn("坐下", replaced[0]["content"])
            self.assertIn("转圈", replaced[0]["content"])
            self.assertEqual(replaced[0]["payload_json"]["metadata"]["action_chain_count"], 2)
        finally:
            manager.close()

    def test_single_action_sequence_creates_action_memory(self):
        manager = self.make_manager()
        try:
            manager.add_conversation_turn(
                "single-chain",
                "user-001",
                "dog-001",
                "坐下",
                "好的",
                timestamp="2026-07-02T09:00:00+08:00",
                model_event_routes=[
                    {
                        "type": "action_sequence",
                        "decision": "create",
                        "confidence": 0.9,
                        "actions": [{"code": "sit", "label_zh": "坐下"}],
                    }
                ],
            )
            manager.trigger_daily_event_extraction("user-001", "dog-001", "2026-07-02")
            memories = manager.events.list_action_memories("user-001", "dog-001", memory_date="2026-07-02")
            self.assertEqual(len(memories), 1)
            self.assertEqual(memories[0]["event_type"], "action_memory")
            self.assertIn("坐下", memories[0]["content"])
        finally:
            manager.close()

    def test_event_text_api_returns_text_without_payload_json(self):
        manager = self.make_manager()
        router = MemoryDebugRouter(manager, FakeLLM())
        try:
            result = manager.add_conversation_turn(
                "text-action",
                "user-001",
                "dog-001",
                "往前走然后坐下",
                "好的",
                timestamp="2026-07-02T09:00:00+08:00",
                model_event_routes=[
                    {
                        "type": "action_sequence",
                        "decision": "create",
                        "confidence": 0.9,
                        "actions": [
                            {"code": "forward", "label_zh": "往前走"},
                            {"code": "sit", "label_zh": "坐下"},
                        ],
                    }
                ],
            )
            manager.trigger_daily_event_extraction("user-001", "dog-001", "2026-07-02")
            texts = router.event_texts(
                "user-001",
                "dog-001",
                "action_memory",
                result["session_id"],
                "2026-07-02",
            )["memories"]
            self.assertEqual(len(texts), 1)
            self.assertIn("往前走 -> 坐下", texts[0]["text"])
            self.assertNotIn("payload_json", texts[0])
        finally:
            manager.close()

    def test_event_text_api_is_action_only(self):
        manager = self.make_manager()
        router = MemoryDebugRouter(manager, FakeLLM())
        try:
            manager.add_conversation_turn(
                "existing-action",
                "user-001",
                "dog-001",
                "坐下",
                "好的",
                timestamp="2026-07-02T09:00:00+08:00",
                model_event_routes=[
                    {
                        "type": "action_sequence",
                        "decision": "create",
                        "confidence": 0.9,
                        "actions": [{"code": "sit", "label_zh": "坐下"}],
                    }
                ],
            )
            manager.trigger_daily_event_extraction("user-001", "dog-001", "2026-07-02")
            self.assertEqual(len(router.event_texts("user-001", "dog-001", "action_memory", None, None)["memories"]), 1)
            texts = router.event_texts("user-001", "dog-001", "message", None, None)["memories"]
            self.assertEqual(texts, [])
        finally:
            manager.close()

    def test_daily_event_extraction_does_not_create_time_memory(self):
        manager = self.make_manager()
        router = MemoryDebugRouter(manager, FakeLLM())
        try:
            manager.add_conversation_turn(
                "daily-event-only",
                "user-001",
                "dog-001",
                "坐下",
                "好的",
                timestamp="2026-07-02T09:00:00+08:00",
                model_event_routes=[
                    {
                        "type": "action_sequence",
                        "decision": "create",
                        "confidence": 0.9,
                        "actions": [{"code": "sit", "label_zh": "坐下"}],
                    }
                ],
            )
            result = router.extract_daily_events("user-001", "dog-001", "2026-07-02")
            self.assertEqual(result["process"]["succeeded"], 1)
            self.assertEqual(result["event_memory_count"], 1)
            self.assertEqual(len(result["event_memories"]), 1)
            self.assertIn("事件链路", result["event_memories"][0]["text"])
            self.assertEqual(manager.events.list_time_memories("user-001", "dog-001"), [])
        finally:
            manager.close()

    def test_action_feedback_route_merges_with_recent_action_memory(self):
        manager = self.make_manager()
        try:
            action = manager.add_conversation_turn(
                "feedback-action-1",
                "user-001",
                "dog-001",
                "坐下",
                "收到",
                timestamp="2026-07-02T09:00:00+08:00",
                model_event_routes=[
                    {
                        "type": "action_sequence",
                        "decision": "create",
                        "confidence": 0.9,
                        "actions": [{"code": "sit", "label_zh": "坐下"}],
                    }
                ],
            )
            result = manager.add_conversation_turn(
                "feedback-action-2",
                "user-001",
                "dog-001",
                "你做得太棒了",
                "谢谢",
                timestamp="2026-07-02T09:00:05+08:00",
                session_id=action["session_id"],
                model_event_routes=[
                    {
                        "type": "action_feedback",
                        "decision": "create",
                        "confidence": 0.9,
                        "feedback": "用户表示肯定",
                    }
                ],
            )
            self.assertIn("action_feedback_event_id", result)
            manager.trigger_daily_event_extraction("user-001", "dog-001", "2026-07-02")
            memories = manager.events.list_action_memories("user-001", "dog-001", memory_date="2026-07-02")
            self.assertEqual(len(memories), 1)
            self.assertIn("事件记忆", memories[0]["content"])
            self.assertIn("事件链路", memories[0]["content"])
            self.assertIn("坐下", memories[0]["content"])
            self.assertIn("用户表示肯定", memories[0]["content"])
            self.assertEqual(
                [item["code"] for item in memories[0]["payload_json"]["actions"]],
                ["sit"],
            )
            self.assertEqual(memories[0]["event_type"], "action_memory")
        finally:
            manager.close()

    def test_action_feedback_without_previous_action_is_ignored(self):
        manager = self.make_manager()
        try:
            result = manager.add_conversation_turn(
                "feedback-without-action",
                "user-001",
                "dog-001",
                "你做得太棒了",
                "谢谢",
                timestamp="2026-07-02T09:00:00+08:00",
                model_event_routes=[
                    {
                        "type": "action_feedback",
                        "decision": "create",
                        "confidence": 0.9,
                        "feedback": "用户表示肯定",
                    }
                ],
            )
            self.assertNotIn("action_feedback_event_id", result)
            manager.trigger_daily_event_extraction("user-001", "dog-001", "2026-07-02")
            memories = manager.events.list_action_memories("user-001", "dog-001", memory_date="2026-07-02")
            self.assertEqual(memories, [])
        finally:
            manager.close()

    def test_action_feedback_does_not_cross_session_boundary(self):
        manager = self.make_manager()
        try:
            manager.add_conversation_turn(
                "feedback-cross-session-action",
                "user-001",
                "dog-001",
                "坐下",
                "收到",
                timestamp="2026-07-02T09:00:00+08:00",
                model_event_routes=[
                    {
                        "type": "action_sequence",
                        "decision": "create",
                        "confidence": 0.9,
                        "actions": [{"code": "sit", "label_zh": "坐下"}],
                    }
                ],
            )
            result = manager.add_conversation_turn(
                "feedback-cross-session-praise",
                "user-001",
                "dog-001",
                "你做得太棒了",
                "谢谢",
                timestamp="2026-07-02T09:00:20+08:00",
                model_event_routes=[
                    {
                        "type": "action_feedback",
                        "decision": "create",
                        "confidence": 0.9,
                        "feedback": "用户表示肯定",
                    }
                ],
            )
            self.assertNotIn("action_feedback_event_id", result)
            manager.trigger_daily_event_extraction("user-001", "dog-001", "2026-07-02")
            memories = manager.events.list_action_memories("user-001", "dog-001", memory_date="2026-07-02")
            self.assertEqual(len(memories), 1)
            self.assertNotIn("用户表示肯定", memories[0]["content"])
        finally:
            manager.close()

    def test_session_detail_filters_messages_and_summary_by_session(self):
        manager = self.make_manager()
        router = MemoryDebugRouter(manager, FakeLLM())
        try:
            first = manager.add_conversation_turn(
                "session-a",
                "user-001",
                "dog-001",
                "第一段会话",
                "第一段回复",
                timestamp="2026-07-02T09:00:00+08:00",
            )
            second = manager.add_conversation_turn(
                "session-b",
                "user-001",
                "dog-001",
                "第二段会话",
                "第二段回复",
                timestamp="2026-07-02T09:00:16+08:00",
            )
            detail = router.session_detail("user-001", second["session_id"])
            self.assertEqual(detail["session_id"], second["session_id"])
            self.assertTrue(all(item["session_id"] == second["session_id"] for item in detail["messages"]))
            self.assertIsNone(detail["summary"])
            sessions = router.sessions("user-001", "dog-001")["sessions"]
            self.assertTrue(any(item["session_id"] == first["session_id"] for item in sessions))
            self.assertTrue(any(item["session_id"] == second["session_id"] for item in sessions))
        finally:
            manager.close()

    def test_session_detail_summary_uses_real_rolling_summary_after_ten_turns(self):
        manager = self.make_manager()
        manager.summarizer = CaptureSummarizer()
        router = MemoryDebugRouter(manager, FakeLLM())
        try:
            first = manager.add_conversation_turn(
                "roll-0",
                "user-001",
                "dog-001",
                "问题0",
                "回答0",
                timestamp="2026-07-02T09:00:00+08:00",
                prompt_token_count=6000,
            )
            self.assertIsNone(router.session_detail("user-001", first["session_id"])["summary"])
            for idx in range(1, 20):
                manager.add_conversation_turn(
                    f"roll-{idx}",
                    "user-001",
                    "dog-001",
                    f"问题{idx}",
                    f"回答{idx}",
                    timestamp=f"2026-07-02T09:00:{idx:02d}+08:00",
                    session_id=first["session_id"],
                    prompt_token_count=6000,
                )
            self.assertTrue(manager.wait_for_summaries())
            detail = router.session_detail("user-001", first["session_id"])
            self.assertIsNotNone(detail["summary"])
            self.assertIn("summary:", detail["summary"]["summary_text"])
            self.assertIn("compacted_through_event_id", detail["summary"])
        finally:
            manager.close()

    def test_weekly_action_memories_create_action_preference_memory_not_longterm_pref(self):
        manager = self.make_manager()
        manager.preference_extractor = FakeExtractor(
            action_result={
                "schema_version": "1.0",
                "user_id": "user-001",
                "memories": [
                    {
                        "content": "七天动作偏好记忆（2026-06-30 至 2026-07-02）：用户多次要求回家先开灯再播放新闻。",
                        "title": "回家动作偏好",
                        "confidence": 0.91,
                        "source_event_ids": [1, 2],
                        "reason_zh": "七天内多次出现同类动作链路",
                    }
                ],
            }
        )
        try:
            for idx, day in enumerate(("2026-06-30", "2026-07-02"), start=1):
                manager.add_conversation_turn(
                    f"weekly-action-{idx}",
                    "user-001",
                    "dog-001",
                    "回家先开灯再播放新闻",
                    "好的",
                    timestamp=f"{day}T09:00:00+08:00",
                    model_event_routes=[
                        {
                            "type": "action_sequence",
                            "decision": "create",
                            "confidence": 0.9,
                            "actions": [
                                {"code": "turn_on_light", "label_zh": "开灯"},
                                {"code": "play_news", "label_zh": "播放新闻"},
                            ],
                        }
                    ],
                )
                manager.trigger_daily_event_extraction("user-001", "dog-001", day)
            manager.add_conversation_turn(
                "weekly-non-repeat",
                "user-001",
                "dog-001",
                "坐下",
                "好的",
                timestamp="2026-07-01T09:00:00+08:00",
                model_event_routes=[
                    {
                        "type": "action_sequence",
                        "decision": "create",
                        "confidence": 0.9,
                        "actions": [{"code": "sit", "label_zh": "坐下"}],
                    }
                ],
            )
            manager.trigger_daily_event_extraction("user-001", "dog-001", "2026-07-01")
            self.assertEqual(len(manager.events.list_action_memories("user-001", "dog-001")), 3)
            result = manager.trigger_weekly_action_preference_extraction(
                "user-001",
                "dog-001",
                "2026-07-02",
            )
            self.assertFalse(result["created_job"])
            self.assertEqual(result["process"]["claimed"], 1)
            self.assertEqual(result["process"]["succeeded"], 1)
            self.assertEqual(result["process"]["jobs"][0]["stored_action_preference_memories"], 1)
            self.assertEqual(len(manager.preference_extractor.action_calls), 1)
            context = manager.preference_extractor.action_calls[0]["action_memory_context"]
            self.assertEqual(context["context_mode"], "seven_day_action_memory_text")
            self.assertIn("事件记忆（2026-06-30）", context["combined_text"])
            prefs = manager.events.list_preferences("user-001", status=None)
            self.assertEqual(prefs, [])
            action_prefs = manager.events.list_action_preference_memories("user-001", "dog-001", end_date="2026-07-02")
            self.assertEqual(len(action_prefs), 1)
            self.assertEqual(action_prefs[0]["event_type"], "action_preference_memory")
            self.assertIn("多次要求回家先开灯再播放新闻", action_prefs[0]["content"])
        finally:
            manager.close()

    def test_weekly_action_preference_skips_when_model_returns_empty(self):
        manager = self.make_manager()
        manager.preference_extractor = FakeExtractor()
        try:
            manager.add_conversation_turn(
                "weekly-empty-model",
                "user-001",
                "dog-001",
                "坐下",
                "好的",
                timestamp="2026-07-02T09:00:00+08:00",
                model_event_routes=[
                    {
                        "type": "action_sequence",
                        "decision": "create",
                        "confidence": 0.9,
                        "actions": [{"code": "sit", "label_zh": "坐下"}],
                    }
                ],
            )
            manager.trigger_daily_event_extraction("user-001", "dog-001", "2026-07-02")
            result = manager.trigger_weekly_action_preference_extraction("user-001", "dog-001", "2026-07-02")
            self.assertEqual(result["process"]["skipped"], 1)
            self.assertEqual(result["process"]["jobs"][0]["model_extracted_memories"], 0)
            self.assertEqual(manager.events.list_action_preference_memories("user-001", "dog-001"), [])
        finally:
            manager.close()

    def test_monday_scheduler_waits_for_daily_action_memory_then_queues_natural_week(self):
        manager = self.make_manager()
        manager.preference_extractor = FakeExtractor()
        try:
            manager.add_conversation_turn(
                "scheduled-weekly-action",
                "user-001",
                "dog-001",
                "坐下",
                "好的",
                timestamp="2026-07-02T09:00:00+08:00",
                model_event_routes=[
                    {
                        "type": "action_sequence",
                        "decision": "create",
                        "confidence": 0.9,
                        "actions": [{"code": "sit", "label_zh": "坐下"}],
                    }
                ],
            )
            first = manager.schedule_due_memory_jobs(
                datetime(2026, 7, 6, 1, 0, tzinfo=timezone(timedelta(hours=8)))
            )
            self.assertEqual(first["weekly_created"], [])
            process_daily = manager.process_memory_jobs_once(limit=20, include_daily=True)
            self.assertEqual(process_daily["succeeded"], 2)
            second = manager.schedule_due_memory_jobs(
                datetime(2026, 7, 6, 1, 1, tzinfo=timezone(timedelta(hours=8)))
            )
            self.assertEqual(len(second["weekly_created"]), 1)
            weekly = second["weekly_created"][0]
            self.assertEqual(weekly["start_date"], "2026-06-29")
            self.assertEqual(weekly["end_date"], "2026-07-05")
            process_weekly = manager.process_memory_jobs_once(limit=20, include_daily=True)
            self.assertEqual(process_weekly["succeeded"], 1)
            self.assertEqual(process_weekly["skipped"], 1)
            duplicate = manager.schedule_due_memory_jobs(
                datetime(2026, 7, 6, 2, 0, tzinfo=timezone(timedelta(hours=8)))
            )
            self.assertEqual(duplicate["weekly_created"], [])
            weekly_jobs = manager.events.list_jobs(
                "user-001", "weekly_action_preference_extract", limit=20
            )
            self.assertEqual(len(weekly_jobs), 1)
            self.assertEqual(weekly_jobs[0]["period_start"], "2026-06-29")
            self.assertEqual(weekly_jobs[0]["period_end"], "2026-07-05")
        finally:
            manager.close()

    def test_model_action_sequence_enters_event_library_in_order(self):
        manager = self.make_manager()
        try:
            manager.add_conversation_turn(
                "r-action",
                "user-001",
                "dog-001",
                "往前走往后走往左走然后坐下",
                "好的",
                model_event_routes=[
                    {
                        "type": "action_sequence",
                        "decision": "create",
                        "confidence": 0.9,
                        "actions": [
                            {"code": "forward", "label_zh": "往前走"},
                            {"code": "backward", "label_zh": "往后走"},
                            {"code": "left", "label_zh": "往左走"},
                            {"code": "sit", "label_zh": "坐下"},
                        ],
                    }
                ],
            )
            events = manager.events.list_action_events("user-001", "dog-001")
            self.assertEqual(
                [item["code"] for item in events[0]["payload_json"]["actions"]],
                ["forward", "backward", "left", "sit"],
            )
        finally:
            manager.close()

    def test_latest_action_context_comes_from_model_created_events(self):
        manager = self.make_manager()
        llm = FakeLLM()
        router = MemoryDebugRouter(manager, llm)
        try:
            manager.add_conversation_turn(
                "ctx-action",
                "user-001",
                "dog-001",
                "执行动作",
                "好的",
                model_event_routes=[
                    {
                        "type": "action_sequence",
                        "decision": "create",
                        "confidence": 0.9,
                        "actions": [{"code": "sit", "label_zh": "坐下"}],
                    }
                ],
            )
            router.submit("user-001", "dog-001", "重复上次操作", debug=True)
            latest = llm.calls[-1]["latest_action_sequence"]
            self.assertIsNotNone(latest)
            self.assertEqual(latest["event_type"], "action_sequence")
        finally:
            manager.close()

    def test_strict_redis_raises_instead_of_silent_fallback(self):
        memory = ShortTermMemory(redis_client=BrokenRedisClient(), allow_memory_fallback=False)
        with self.assertRaises(ConnectionError):
            memory.append_conversation("d", "u", [{"content": "x"}])

    def test_ui_debug_apis_are_reachable(self):
        manager = self.make_manager()
        router = MemoryDebugRouter(manager, FakeLLM())
        try:
            router.submit("u", "d", "你好", debug=True)
            latest_user = manager.events.list_events(user_id="u", device_id="d", role="user", limit=1)[0]
            memory_date = manager._local_date(latest_user["created_at"])
            router.extract_daily_memory("u", "d", memory_date)
            router.update_debug_device_state("d", {"battery": 80}, "2026-06-25T10:00:00+08:00")
            manager.add_conversation_turn(
                "ui-action",
                "u",
                "d",
                "往前走然后坐下",
                "好的",
                model_event_routes=[
                    {
                        "type": "action_sequence",
                        "decision": "create",
                        "confidence": 0.9,
                        "actions": [
                            {"code": "forward", "label_zh": "往前走"},
                            {"code": "sit", "label_zh": "坐下"},
                        ],
                    }
                ],
            )
            router.extract_daily_events("u", "d", memory_date)
            self.assertIn("user_card", router.debug_user("u"))
            self.assertEqual(len(router.time_memories("u", "d")["time_memories"]), 1)
            self.assertEqual(len(router.action_events("u", "d")["actions"]), 1)
            self.assertTrue(router.event_library("u", "d", "action_memory")["events"])
            self.assertTrue(router.event_library("u", "d", None)["events"])
            self.assertIn("events", router.events("u", None, None))
            self.assertIn("state", router.debug_device("d"))
            self.assertTrue(router.status()["ready"])
        finally:
            manager.close()

    def test_jsonl_migration_uses_legacy_unassigned(self):
        path = Path(self.temp.name) / "legacy.jsonl"
        path.write_text(
            json.dumps(
                {
                    "id": "old-1",
                    "session_id": "legacy-device",
                    "type": "conversation",
                    "content": [{"role": "user", "content": "旧记忆"}],
                    "timestamp": "2025-01-01T00:00:00+00:00",
                    "metadata": {},
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        manager = self.make_manager()
        try:
            result = migrate_jsonl(manager.events, path)
            self.assertEqual(result["imported"], 1)
            events = manager.events.list_events(user_id=LEGACY_USER_ID, event_type="legacy_jsonl")
            self.assertEqual(events[0]["user_id"], LEGACY_USER_ID)
            self.assertEqual(events[0]["device_id"], "legacy-device")
        finally:
            manager.close()

    def test_old_sqlite_migration_does_not_assign_real_user(self):
        db = Path(self.temp.name) / "old.db"
        legacy_index_table = "index" + "_" + "outbox"
        con = sqlite3.connect(db)
        con.executescript(
            f"""
            CREATE TABLE events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL,
                archived_at TEXT
            );
            CREATE TABLE {legacy_index_table} (id INTEGER PRIMARY KEY AUTOINCREMENT, memory_id TEXT);
            INSERT INTO events(session_id,event_type,payload,created_at)
            VALUES('dog-old','message','{{"role":"user","content":"旧设备记忆"}}','2025-01-01T00:00:00+00:00');
            """
        )
        con.commit()
        con.close()
        store = SQLiteEventStore(db)
        try:
            events = store.list_events(user_id=LEGACY_USER_ID)
            self.assertEqual(events[0]["device_id"], "dog-old")
            tables = [
                row[0]
                for row in store._conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            ]
            self.assertNotIn(legacy_index_table, tables)
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
