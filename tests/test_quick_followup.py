"""快捷指令胶囊与追问意图验收测试（quick_followup_fix tasks.md 任务组 8）。

TestFollowupRoute：追问路由拦截/排除抽样/确定性/兜底（LLM 替身计数）；
TestQuickCommands：前端源码锚点（胶囊常量/淡出/溢出合并/请求体扩展）。
"""

from pathlib import Path
from typing import Optional

import pytest
from fastapi.testclient import TestClient

import app.data.memory as memory
from app.main import create_app
from app.rules.agent_router import create_default_agent_router
from app.rules.chat_schemas import (
    FOLLOWUP_EXCLUDED_NOTICE,
    FOLLOWUP_EXHAUSTED_NOTICE,
    FOLLOWUP_NO_CONTEXT_SCRIPT,
    FollowupContext,
)
from app.rules.intent_parser import IntentParser
from app.rules.rec_schemas import Scene
from app.rules.exercise import create_default_exercise_engine


@pytest.fixture(autouse=True)
def isolated_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "MEMORY_FILE_PATH", tmp_path / "memory.json")
    yield


class _LLMCounter:
    """LLM 替身：只计数不回答（追问场景必须零调用）。"""

    def __init__(self):
        self.call_count = 0

    def is_configured(self) -> bool:
        return True

    def chat_with_llm(self, text: str, memory_context: str, history_messages=None) -> str:
        self.call_count += 1
        return "不应该被调用"


def _router_with_llm_counter() -> tuple:
    router = create_default_agent_router()
    counter = _LLMCounter()
    router._llm_client = counter
    return router, counter


def _first_round_shown(router, user_id: str, text: str, limit: int = 3) -> list:
    response = router.route(user_id, text)
    return [a.action_id for a in response.exercise_recommendations.recommended_actions[:limit]]


class TestFollowupRoute:
    """追问路由 10 锚点。"""

    def test_followup_intercepts_before_llm(self):
        router, counter = _router_with_llm_counter()
        shown = _first_round_shown(router, "u1", "宿舍怎么练")
        ctx = FollowupContext(last_text="宿舍怎么练", shown_action_ids=tuple(shown))
        response = router.route("u1", "还有其他什么动作", ctx)
        assert response.route.value == "HEALTH_QUERY"
        assert counter.call_count == 0
        new_ids = [a.action_id for a in response.exercise_recommendations.recommended_actions]
        assert 0 < len(new_ids) <= 3
        assert not (set(new_ids) & set(shown))

    def test_no_context_guidance(self):
        router, counter = _router_with_llm_counter()
        response = router.route("u1", "还有其他什么动作")
        assert response.reply_text == FOLLOWUP_NO_CONTEXT_SCRIPT
        assert counter.call_count == 0
        assert response.diet_recommendations.recommended_dishes == []
        assert response.exercise_recommendations.recommended_actions == []

    def test_pool_exhausted(self):
        router, _ = _router_with_llm_counter()
        full = create_default_exercise_engine().recommend([], Scene.DORM_QUIET, 15, None)
        full_ids = tuple(a.action_id for a in full.recommended_actions)
        ctx = FollowupContext(last_text="宿舍怎么练", shown_action_ids=full_ids)
        response = router.route("u5", "还有其他什么动作", ctx)
        assert response.exercise_recommendations.recommended_actions == []
        assert any("都展示过了" in n for n in response.notices)

    def test_deterministic_sampling(self):
        router, _ = _router_with_llm_counter()
        shown = _first_round_shown(router, "u2", "宿舍怎么练")
        ctx = FollowupContext(last_text="宿舍怎么练", shown_action_ids=tuple(shown))
        first = router.route("u2", "还有其他什么动作", ctx)
        second = router.route("u2", "还有其他什么动作", ctx)
        ids_first = [a.action_id for a in first.exercise_recommendations.recommended_actions]
        ids_second = [a.action_id for a in second.exercise_recommendations.recommended_actions]
        assert ids_first == ids_second

    def test_followup_priority_over_regular_intent(self):
        router, _ = _router_with_llm_counter()
        ctx = FollowupContext(last_text="尿酸高，午饭吃什么", shown_dish_ids=("D001", "D002"))
        response = router.route("u3", "还有别的菜品推荐", ctx)
        assert response.route.value == "HEALTH_QUERY"
        ids = [d.dish_id for d in response.diet_recommendations.recommended_dishes]
        assert ids
        assert "D001" not in ids and "D002" not in ids

    def test_mood_priority_over_followup(self):
        router, _ = _router_with_llm_counter()
        ctx = FollowupContext(last_text="宿舍怎么练", shown_action_ids=("A001",))
        response = router.route("u6", "心情低落好累，再推荐几个", ctx)
        assert response.route.value == "MOOD_SUPPORT"

    def test_zenme_lian_word_hits_exercise_intent(self):
        parser = IntentParser()
        result = parser.parse("宿舍怎么练")
        assert result.exercise_intent is True
        assert result.scene is Scene.DORM_QUIET

    def test_small_talk_regression_untouched(self):
        from app.rules.chat_schemas import SMALL_TALK_SCRIPT

        router = create_default_agent_router()
        response = router.route("u7", "今天天气不错")
        assert response.route.value == "SMALL_TALK"
        assert response.reply_text == SMALL_TALK_SCRIPT

    def test_followup_flag_parsing(self):
        parser = IntentParser()
        assert parser.parse("还有其他什么动作").followup_intent is True
        assert parser.parse("午饭吃什么").followup_intent is False
        fu = parser.parse("还有其他什么动作")
        assert fu.diet_intent is False and fu.exercise_intent is False

    def test_excluded_notice_and_memory_filter(self):
        router, _ = _router_with_llm_counter()
        memory.add_memory("u8", "diet_exclusions", "豆腐")
        ctx = FollowupContext(last_text="午饭吃什么", shown_dish_ids=("D062",))
        response = router.route("u8", "再推荐几个菜", ctx)
        dishes = response.diet_recommendations.recommended_dishes
        assert dishes
        assert "D062" not in {d.dish_id for d in dishes}
        assert all("豆腐" not in d.name for d in dishes)
        assert any("避开刚才展示过的" in n for n in response.notices)


@pytest.fixture(scope="module")
def index_html():
    client = TestClient(create_app())
    response = client.get("/")
    assert response.status_code == 200
    return response.text


class TestQuickCommands:
    """快捷指令胶囊与前端改造 9 锚点。"""

    def test_quick_commands_constant(self, index_html):
        assert "const QUICK_COMMANDS = [" in index_html
        for text in ("食堂吃什么", "宿舍怎么练", "心情不好", "识别体检单"):
            assert text in index_html
        assert "special: 'camera'" in index_html

    def test_send_text_adaptation(self, index_html):
        assert "sendText: '最近心情不好，有点难受'" in index_html
        assert "btn.dataset.q = cmd.sendText;" in index_html

    def test_camera_branch_triggers_file_input(self, index_html):
        assert "cmd.special === 'camera'" in index_html
        assert "document.getElementById('food-file-input').click();" in index_html

    def test_dock_persistence_mechanism(self, index_html):
        assert "function fadeOutQuickCommands" not in index_html
        assert "fadeOutQuickCommands();" not in index_html
        script_start = index_html.find("<script>")
        script_end = index_html.find("</script>")
        script_zone = index_html[script_start:script_end]
        assert "pointer-events-none" not in script_zone
        assert "opacity-0" not in script_zone
        assert 'id="quick-dock"' in index_html
        assert index_html.find("<footer") < index_html.find('id="quick-dock"') < index_html.find('id="btn-send"')

    def test_overflow_hint_merged_single_if(self, index_html):
        assert "if (hasMoreDishes || hasMoreAvoided) {" in index_html
        assert "if (hasMoreActions || hasMoreAvoided) {" in index_html
        assert "if (hasMoreDishes) {" not in index_html
        assert "if (hasMoreActions) {" not in index_html

    def test_shown_ids_collection_in_render_loops(self, index_html):
        assert "shown_dish_ids.add(dish.dish_id);" in index_html
        assert "shown_action_ids.add(action.action_id);" in index_html

    def test_request_body_five_fields(self, index_html):
        assert "last_text: last_user_text || null," in index_html
        assert "shown_dish_ids: Array.from(shown_dish_ids)," in index_html
        assert "shown_action_ids: Array.from(shown_action_ids)" in index_html

    def test_old_quick_ask_removed(self, index_html):
        assert "quick-ask" not in index_html
        assert "尿酸高，午饭吃什么？" not in index_html

    def test_existing_features_kept(self, index_html):
        for anchor in (
            'id="btn-selfcheck"',
            'id="btn-health-form"',
            'id="food-file-input"',
            "function appendMoreHint",
            "function buildQuickCommands",
        ):
            assert anchor in index_html

    def test_chat_api_five_fields_e2e(self):
        """API 级：携带上轮上下文的 chat 请求被正确消费。"""
        client = TestClient(create_app())
        first = client.post(
            "/api/v1/chat",
            json={"user_id": "u-api", "text": "宿舍怎么练"},
        )
        assert first.status_code == 200
        shown = [a["action_id"] for a in first.json()["exercise_recommendations"]["recommended_actions"][:3]]
        second = client.post(
            "/api/v1/chat",
            json={
                "user_id": "u-api",
                "text": "还有其他什么动作",
                "last_text": "宿舍怎么练",
                "shown_action_ids": shown,
            },
        )
        assert second.status_code == 200
        body = second.json()
        assert body["route"] == "HEALTH_QUERY"
        new_ids = [a["action_id"] for a in body["exercise_recommendations"]["recommended_actions"]]
        assert new_ids
        assert not (set(new_ids) & set(shown))

class _HistoryCaptureLLM:
    """三参捕获替身：记录 LLM 兜底收到的历史第三参。"""

    def __init__(self, reply="兜底回复内容"):
        self.reply = reply
        self.call_count = 0
        self.history_received = "unset"
        self.memory_received = None

    def is_configured(self):
        return True

    def chat_with_llm(self, text, memory_context, history_messages=None):
        self.call_count += 1
        self.memory_received = memory_context
        self.history_received = history_messages
        return self.reply


class TestMultiTurnHistory:
    """8.3 端点集成锚点：多轮历史写入与 LLM 注入（multi_turn_identity）。"""

    def _client_with_stub(self):
        application = create_app()
        stub = _HistoryCaptureLLM()
        application.state.agent_router._llm_client = stub
        return TestClient(application), stub

    def test_second_turn_llm_receives_first_turn_history(self):
        client, stub = self._client_with_stub()
        client.post("/api/v1/chat", json={"user_id": "mt1", "text": "徐州有什么好玩的？"})
        client.post("/api/v1/chat", json={"user_id": "mt1", "text": "那有什么好吃的？"})
        assert stub.call_count == 2
        assert isinstance(stub.history_received, list)
        contents = [entry["content"] for entry in stub.history_received]
        assert "徐州有什么好玩的？" in contents
        assistant_turns = [c for c in contents if c.startswith("兜底回复内容")]
        assert assistant_turns
        assert "（本建议仅供日常健康管理参考）" in assistant_turns[0]

    def test_m4_crisis_never_written_to_history(self):
        client, _ = self._client_with_stub()
        response = client.post("/api/v1/chat", json={"user_id": "mt2", "text": "我想死"})
        assert response.status_code == 200
        assert response.json()["route"] == "M4_CRISIS"
        assert memory.get_chat_history("mt2") == []

    def test_rule_engine_turn_also_recorded(self):
        client, _ = self._client_with_stub()
        client.post("/api/v1/chat", json={"user_id": "mt3", "text": "尿酸高，午饭吃什么？"})
        history = memory.get_chat_history("mt3")
        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "尿酸高，午饭吃什么？"}
        assert history[1]["role"] == "assistant"
        assert "为你整理了" in history[1]["content"]

    def test_history_persisted_reread(self, tmp_path, monkeypatch):
        """重启模拟基线：写入后经模块函数重读同一物理文件。"""
        target = tmp_path / "persist.json"
        monkeypatch.setattr(memory, "MEMORY_FILE_PATH", target)
        client, _ = self._client_with_stub()
        client.post("/api/v1/chat", json={"user_id": "mt4", "text": "午饭吃什么？"})
        monkeypatch.setattr(memory, "MEMORY_FILE_PATH", target)
        history = memory.get_chat_history("mt4")
        assert len(history) == 2
        assert history[0]["content"] == "午饭吃什么？"

    def test_history_write_failure_silent(self, monkeypatch):
        def broken_save(user_id, history_list):
            raise OSError("disk full")

        monkeypatch.setattr(memory, "save_chat_history", broken_save)
        import app.api.chat as chat_module

        monkeypatch.setattr(chat_module, "save_chat_history", broken_save)
        client, _ = self._client_with_stub()
        response = client.post("/api/v1/chat", json={"user_id": "mt5", "text": "午饭吃什么？"})
        assert response.status_code == 200
        assert response.json()["route"] == "HEALTH_QUERY"

    def test_history_never_affects_rule_engine(self):
        """核心架构原则验收：带历史与无历史同文本规则响应逐字段一致。"""
        memory.save_chat_history(
            "mt-with",
            [
                {"role": "user", "content": "旧话题一"},
                {"role": "assistant", "content": "旧回答一"},
                {"role": "user", "content": "旧话题二"},
                {"role": "assistant", "content": "旧回答二"},
            ],
        )
        client, _ = self._client_with_stub()
        with_history = client.post(
            "/api/v1/chat", json={"user_id": "mt-with", "text": "尿酸高，午饭吃什么？"}
        ).json()
        without_history = client.post(
            "/api/v1/chat", json={"user_id": "mt-without", "text": "尿酸高，午饭吃什么？"}
        ).json()
        assert with_history["diet_recommendations"] == without_history["diet_recommendations"]
        assert with_history["exercise_recommendations"] == without_history["exercise_recommendations"]
        assert with_history["route"] == without_history["route"]

    def test_p4_unconfigured_zero_llm_and_rule_untouched(self):
        """未配置态 P4 恒寒暄；规则轮（P1/P2/P3）零 LLM 调用旁证。"""
        client = TestClient(create_app())
        small = client.post("/api/v1/chat", json={"user_id": "mt7", "text": "今天天气不错"})
        assert small.json()["route"] == "SMALL_TALK"

        application = create_app()
        stub = _HistoryCaptureLLM()
        application.state.agent_router._llm_client = stub
        stub_client = TestClient(application)
        for text in ("我想死", "我最近考试压力很大，失眠，有什么办法吗？", "尿酸高，午饭吃什么？"):
            stub_client.post("/api/v1/chat", json={"user_id": "mt8", "text": text})
        assert stub.call_count == 0

class TestHistoryEndpoints:
    """6.5 历史端点集成锚点（quick_dock_history tasks 6.5，11 函数）。"""

    def _client(self):
        return TestClient(create_app())

    def _preset_rounds(self, user_id: str, rounds: int = 3):
        for i in range(rounds):
            history = memory.get_chat_history(user_id) + [
                {"role": "user", "content": f"第{i}轮问"},
                {"role": "assistant", "content": f"第{i}轮答"},
            ]
            memory.save_chat_history(user_id, history)

    def test_get_returns_recent_entries_in_order(self):
        self._preset_rounds("hp1")
        response = self._client().get("/api/v1/history/hp1")
        assert response.status_code == 200
        body = response.json()
        assert body["user_id"] == "hp1"
        assert body["count"] == 6
        contents = [e["content"] for e in body["history"]]
        assert contents[0] == "第0轮问"
        assert contents[-1] == "第2轮答"
        for entry in body["history"]:
            assert set(entry.keys()) == {"role", "content"}

    def test_get_idempotent_zero_side_effect(self):
        self._preset_rounds("hp2")
        client = self._client()
        first = client.get("/api/v1/history/hp2")
        before = memory.MEMORY_FILE_PATH.read_text(encoding="utf-8")
        second = client.get("/api/v1/history/hp2")
        after = memory.MEMORY_FILE_PATH.read_text(encoding="utf-8")
        assert first.json() == second.json()
        assert before == after

    def test_get_new_user_returns_empty(self):
        response = self._client().get("/api/v1/history/hp-new")
        assert response.status_code == 200
        body = response.json()
        assert body["history"] == []
        assert body["count"] == 0

    def test_get_corrupted_file_returns_empty(self, tmp_path, monkeypatch):
        broken = tmp_path / "broken.json"
        broken.write_text("{broken", encoding="utf-8")
        monkeypatch.setattr(memory, "MEMORY_FILE_PATH", broken)
        response = self._client().get("/api/v1/history/hp3")
        assert response.status_code == 200
        assert response.json()["count"] == 0

    def test_get_blank_user_id_422(self):
        response = self._client().get("/api/v1/history/%20")
        assert response.status_code == 422
        assert "请提供用户标识" in str(response.json()["detail"])

    def test_delete_clears_history_keeps_preferences(self):
        memory.add_memory("hp4", "diet_exclusions", "香菜")
        memory.add_memory("hp4", "habits", "跑步")
        self._preset_rounds("hp4")
        response = self._client().delete("/api/v1/history/hp4")
        assert response.status_code == 200
        body = response.json()
        assert body == {"user_id": "hp4", "history": [], "count": 0}
        assert memory.get_chat_history("hp4") == []
        kept = memory.get_memory("hp4")
        assert kept["diet_exclusions"] == ["香菜"]
        assert kept["habits"] == ["跑步"]

    def test_delete_idempotent_for_empty_user(self):
        client = self._client()
        first = client.delete("/api/v1/history/hp-empty")
        second = client.delete("/api/v1/history/hp-empty")
        assert first.status_code == second.status_code == 200
        assert first.json() == second.json()

    def test_delete_write_failure_returns_500(self, monkeypatch):
        import app.api.chat as chat_module

        monkeypatch.setattr(chat_module, "save_chat_history", lambda uid, hist: False)
        response = self._client().delete("/api/v1/history/hp5")
        assert response.status_code == 500
        assert response.json()["detail"] == "服务暂时不可用，请稍后再试"

    def test_delete_blank_user_id_422(self):
        response = self._client().delete("/api/v1/history/%20")
        assert response.status_code == 422
        assert "请提供用户标识" in str(response.json()["detail"])

    def test_m4_crisis_never_in_history_endpoint(self):
        client = self._client()
        client.post("/api/v1/chat", json={"user_id": "hp6", "text": "我想死"})
        client.post("/api/v1/chat", json={"user_id": "hp6", "text": "今天天气不错"})
        body = client.get("/api/v1/history/hp6").json()
        contents = [e["content"] for e in body["history"]]
        assert "我想死" not in contents
        assert "今天天气不错" in contents

    def test_history_endpoint_reads_twenty_entries(self):
        history = [{"role": "user", "content": f"消息{i}"} for i in range(22)]
        memory.save_chat_history("hp7", history)
        body = self._client().get("/api/v1/history/hp7").json()
        assert body["count"] == 20