"""长期记忆 + LLM 兜底 + 情绪便签 UI 验收测试（memory_llm_ui tasks.md 任务组 6）。

七个测试类：存储契约 / 抽取正则 / 偏好过滤 / LLM 客户端 / P4 三级
降级 / 路由编排集成 / 前端便签锚点。全部用 tmp_path 隔离记忆路径
（三纪律之二）、httpx.MockTransport / 协议 Stub 替身，不触碰真实
凭证与工作区 memory.json。
"""

import json
import logging
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

import app.data.memory as memory
import app.utils.llm as llm_module
from app.data.memory import (
    MEMORY_EMPTY,
    MEMORY_KEY_DISPLAY,
    add_memory,
    build_memory_context,
    expand_allergy_terms,
    extract_memory_preferences,
    get_memory,
)
from app.main import create_app
from app.rules.agent_router import create_default_agent_router
from app.rules.chat_schemas import DISCLAIMER_TEXT, SMALL_TALK_SCRIPT
from app.rules.rec_schemas import DietRecommendResult, RecommendedDish
from app.utils.llm import (
    LLMClient,
    LLMClientError,
    LLM_DISCLAIMER_SUFFIX,
    LLM_FALLBACK_SCRIPT,
    build_llm_messages,
    build_system_prompt,
)
from app.data.memory import get_chat_history, get_user_memory, save_chat_history, save_user_memory

ROOT_MEMORY = Path("memory.json")


@pytest.fixture(autouse=True)
def isolated_memory(tmp_path, monkeypatch):
    """三纪律之二：全部用例记忆路径重定向至 tmp_path，零工作区污染。"""
    monkeypatch.setattr(memory, "MEMORY_FILE_PATH", tmp_path / "memory.json")
    yield


class TestMemoryStore:
    """6.1 存储契约与容错。"""

    def test_add_and_get_roundtrip(self):
        assert add_memory("stu001", "diet_exclusions", "香菜") is True
        assert get_memory("stu001") == {
            "diet_exclusions": ["香菜"],
            "allergy_history": [],
            "habits": [],
        }
        store = json.loads(memory.MEMORY_FILE_PATH.read_text(encoding="utf-8"))
        assert store["stu001"]["diet_exclusions"] == ["香菜"]

    def test_missing_file_and_new_user_return_empty(self):
        assert get_memory("nobody") == MEMORY_EMPTY
        assert not memory.MEMORY_FILE_PATH.exists()

    def test_corrupted_file_returns_empty_with_warning(self, caplog):
        memory.MEMORY_FILE_PATH.write_text("{broken json", encoding="utf-8")
        with caplog.at_level(logging.WARNING):
            assert get_memory("stu001") == MEMORY_EMPTY
        assert "记忆文件" in caplog.text

    def test_idempotent_dedup(self):
        assert add_memory("stu001", "diet_exclusions", "香菜") is True
        assert add_memory("stu001", "diet_exclusions", "香菜") is False
        assert get_memory("stu001")["diet_exclusions"] == ["香菜"]

    def test_invalid_key_raises_without_write(self):
        with pytest.raises(ValueError):
            add_memory("stu001", "color", "红色")
        assert not memory.MEMORY_FILE_PATH.exists()

    def test_user_isolation(self):
        add_memory("u1", "habits", "每天跑步")
        add_memory("u2", "allergy_history", "海鲜")
        assert get_memory("u1")["habits"] == ["每天跑步"]
        assert get_memory("u2")["allergy_history"] == ["海鲜"]

    def test_ten_writes_persisted(self):
        for i in range(10):
            assert add_memory("stu001", "habits", f"习惯{i}") is True
        assert len(get_memory("stu001")["habits"]) == 10

    def test_memory_keys_display(self):
        assert MEMORY_KEY_DISPLAY["diet_exclusions"] == "忌口"
        assert MEMORY_KEY_DISPLAY["allergy_history"] == "过敏史"
        assert MEMORY_KEY_DISPLAY["habits"] == "习惯"


class TestMemoryExtraction:
    """6.2 抽取正则契约。"""

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("我不吃香菜", [("diet_exclusions", "香菜")]),
            ("我不喝牛奶", [("diet_exclusions", "牛奶")]),
            ("我对海鲜过敏", [("allergy_history", "海鲜")]),
            ("我花生过敏", [("allergy_history", "花生")]),
            ("我习惯每天跑步", [("habits", "每天跑步")]),
            ("我喜欢睡前拉伸", [("habits", "睡前拉伸")]),
            ("今天天气不错", []),
        ],
    )
    def test_extraction_patterns(self, text, expected):
        assert extract_memory_preferences(text) == expected

    def test_long_allergy_shields_short_mode(self):
        assert extract_memory_preferences("我对海鲜过敏") == [("allergy_history", "海鲜")]

    def test_multi_hit_in_one_text(self):
        hits = extract_memory_preferences("我不吃香菜，我对海鲜过敏")
        assert ("diet_exclusions", "香菜") in hits
        assert ("allergy_history", "海鲜") in hits

    def test_build_memory_context(self):
        ctx = build_memory_context(
            {"diet_exclusions": ["香菜"], "allergy_history": [], "habits": ["每天跑步"]}
        )
        assert ctx == "忌口：香菜；习惯：每天跑步"
        assert build_memory_context(MEMORY_EMPTY) == ""

    def test_expand_allergy_terms(self):
        assert expand_allergy_terms(["乳糖不耐"]) == ("乳糖不耐", "乳", "奶", "牛奶", "酸奶", "奶油")
        assert expand_allergy_terms(["未收录原"]) == ("未收录原",)


def _dish(dish_id, name):
    return RecommendedDish(dish_id=dish_id, name=name, reason="测试")


class TestMemoryFilters:
    """6.3 偏好后过滤契约（经 _apply_memory_filters 直测，dish_id 取真实菜品）。"""

    def _make_result(self):
        return DietRecommendResult(
            recommended_dishes=[
                _dish("D001", "麻婆豆腐"),
                _dish("D062", "啤酒鸭"),
                _dish("D002", "家常豆腐"),
            ],
        )

    def test_diet_exclusion_filters_by_name(self):
        from app.rules.agent_router import _apply_memory_filters

        result = self._make_result()
        filtered, notices = _apply_memory_filters(result, {"diet_exclusions": ["豆腐"], "allergy_history": [], "habits": []})
        assert [d.name for d in filtered.recommended_dishes] == ["啤酒鸭"]
        assert "已根据你的忌口偏好（不吃豆腐）过滤推荐菜品" in notices

    def test_allergy_filters_by_expansion_name_hit(self):
        from app.rules.agent_router import _apply_memory_filters

        result = self._make_result()
        filtered, notices = _apply_memory_filters(result, {"diet_exclusions": [], "allergy_history": ["豆腐"], "habits": []})
        assert [d.name for d in filtered.recommended_dishes] == ["啤酒鸭"]
        assert any("过敏史" in n for n in notices)

    def test_no_memory_returns_original(self):
        from app.rules.agent_router import _apply_memory_filters

        result = self._make_result()
        filtered, notices = _apply_memory_filters(result, MEMORY_EMPTY)
        assert filtered is result
        assert notices == []

    def test_all_filtered_notice(self):
        from app.rules.agent_router import _apply_memory_filters

        result = DietRecommendResult(recommended_dishes=[_dish("D001", "麻婆豆腐")])
        filtered, notices = _apply_memory_filters(result, {"diet_exclusions": ["豆腐"], "allergy_history": [], "habits": []})
        assert filtered.recommended_dishes == []
        assert any("全部按你的偏好过滤" in n for n in notices)


class _LLMStub:
    """协议级替身：捕获调用并可编程响应。"""

    def __init__(self, reply=None, raises=False, configured=True):
        self._reply = reply
        self._raises = raises
        self._configured = configured
        self.call_count = 0
        self.last_memory_context = None

    def is_configured(self) -> bool:
        return self._configured

    def chat_with_llm(self, text, memory_context, history_messages=None):
        self.call_count += 1
        self.last_memory_context = memory_context
        if self._raises:
            raise LLMClientError("stub failure")
        return self._reply


class TestLLMClient:
    """6.4 LLM 客户端契约与请求体出境断言。"""

    def _transport(self, captured, content=" 清淡饮食建议 "):
        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content.decode("utf-8"))
            captured["auth"] = request.headers.get("Authorization")
            return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

        return httpx.MockTransport(handler)

    def test_success_strips_whitespace(self):
        captured = {}
        client = LLMClient(transport=self._transport(captured))
        assert client.chat_with_llm("晚饭吃什么", "忌口：香菜") == "清淡饮食建议"

    def test_request_body_shape(self):
        captured = {}
        client = LLMClient(transport=self._transport(captured))
        client.chat_with_llm("晚饭吃什么", "忌口：香菜")
        body = captured["body"]
        assert body["model"] == "glm-4-flash"
        msgs = body["messages"]
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert "忌口：香菜" in msgs[0]["content"]
        assert msgs[1] == {"role": "user", "content": "晚饭吃什么"}
        assert captured["auth"].startswith("Bearer ")
        assert "user_id" not in json.dumps(body)

    def test_empty_memory_context_system_prompt(self):
        captured = {}
        client = LLMClient(transport=self._transport(captured))
        client.chat_with_llm("你好", "")
        assert "个人偏好" not in captured["body"]["messages"][0]["content"]

    def test_non_2xx_raises(self):
        client = LLMClient(transport=httpx.MockTransport(lambda req: httpx.Response(500, json={})))
        with pytest.raises(LLMClientError):
            client.chat_with_llm("x", "")

    def test_bad_structure_raises(self):
        client = LLMClient(transport=httpx.MockTransport(lambda req: httpx.Response(200, json={})))
        with pytest.raises(LLMClientError):
            client.chat_with_llm("x", "")

    def test_build_system_prompt(self):
        assert "个人偏好" in build_system_prompt("忌口：香菜")
        assert "个人偏好" not in build_system_prompt("")

    def test_frozen_constants(self):
        assert LLM_DISCLAIMER_SUFFIX == "（本建议仅供日常健康管理参考）"
        assert LLM_FALLBACK_SCRIPT == "抱歉，这个我还在学习中。你可以问我今天该吃什么，或者去操场该做什么运动。"


class TestLLMFallback:
    """6.5 P4 三级降级链。"""

    def _router(self, llm_stub):
        router = create_default_agent_router()
        router._llm_client = llm_stub
        return router

    def test_no_client_returns_small_talk_script(self):
        response = create_default_agent_router().route("u1", "今天天气不错")
        assert response.reply_text == SMALL_TALK_SCRIPT
        assert response.disclaimer == DISCLAIMER_TEXT

    def test_unconfigured_stub_returns_small_talk_script(self):
        response = self._router(_LLMStub(configured=False)).route("u1", "今天天气不错")
        assert response.reply_text == SMALL_TALK_SCRIPT

    def test_failure_returns_fallback_script(self):
        response = self._router(_LLMStub(raises=True)).route("u1", "量子力学的本质是什么？")
        assert response.reply_text == LLM_FALLBACK_SCRIPT

    def test_arbitrary_exception_returns_fallback(self):
        class Boom:
            def is_configured(self):
                return True

            def chat_with_llm(self, text, memory_context, history_messages=None):
                raise RuntimeError("boom")

        response = self._router(Boom()).route("u1", "量子力学的本质是什么？")
        assert response.reply_text == LLM_FALLBACK_SCRIPT

    def test_success_appends_disclaimer_suffix(self):
        response = self._router(_LLMStub(reply="这是一条通用回答。")).route("u1", "徐州有什么好玩的地方？")
        assert response.reply_text.startswith("这是一条通用回答。")
        assert response.reply_text.endswith(LLM_DISCLAIMER_SUFFIX)
        assert response.disclaimer == DISCLAIMER_TEXT
        assert response.route.value == "SMALL_TALK"

    def test_reply_truncated_to_1000(self):
        response = self._router(_LLMStub(reply="字" * 1500)).route("u1", "讲个长故事")
        content = response.reply_text[: -len("\n" + LLM_DISCLAIMER_SUFFIX)]
        assert len(content) == 1000

    def test_empty_reply_falls_back(self):
        response = self._router(_LLMStub(reply="   ")).route("u1", "你好呀")
        assert response.reply_text.startswith(LLM_FALLBACK_SCRIPT)

    @pytest.mark.parametrize(
        "text",
        ["我想死", "我最近考试压力很大，失眠，有什么办法吗？", "尿酸高，午饭吃什么？"],
    )
    def test_p123_never_call_llm(self, text):
        stub = _LLMStub(reply="回答")
        response = self._router(stub).route("u1", text)
        assert stub.call_count == 0
        assert response.route.value in ("M4_CRISIS", "MOOD_SUPPORT", "HEALTH_QUERY")


class TestRouterMemoryIntegration:
    """6.6 路由编排集成。"""

    def test_m4_never_extracts(self):
        add_memory("u-m4", "diet_exclusions", "基线")  # 预置基线验证零新增
        router = create_default_agent_router()
        response = router.route("u-m4", "我想死，我不吃香菜")
        assert response.route.value == "M4_CRISIS"
        assert get_memory("u-m4")["diet_exclusions"] == ["基线"]

    def test_same_turn_extraction_and_filter(self):
        router = create_default_agent_router()
        response = router.route("u-new", "我不吃豆腐，午饭吃什么？")
        assert response.route.value == "HEALTH_QUERY"
        assert get_memory("u-new")["diet_exclusions"] == ["豆腐"]
        assert any("已记住你的偏好" in n for n in response.notices)
        assert any("忌口偏好" in n for n in response.notices)
        assert all("豆腐" not in d.name for d in response.diet_recommendations.recommended_dishes)

    def test_mood_route_records_preference(self):
        router = create_default_agent_router()
        response = router.route("u-mood", "我最近很孤单，顺便说下我不吃香菜")
        assert response.route.value == "MOOD_SUPPORT"
        assert get_memory("u-mood")["diet_exclusions"] == ["香菜"]

    def test_exercise_query_ignores_diet_memory(self):
        add_memory("u-ex", "diet_exclusions", "香菜")
        router = create_default_agent_router()
        with_memory = router.route("u-ex", "在宿舍怎么运动 10 分钟？")
        baseline = create_default_agent_router().route("u-baseline", "在宿舍怎么运动 10 分钟？")
        assert with_memory.exercise_recommendations.model_dump() == baseline.exercise_recommendations.model_dump()

    def test_recommend_endpoint_unaffected_by_memory(self):
        add_memory("u-api", "diet_exclusions", "豆腐")
        client = TestClient(create_app())
        response = client.post(
            "/api/v1/recommend/diet", json={"tags": ["高尿酸"], "meal": "lunch"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["recommended_dishes"]
        assert all("忌口偏好" not in n for n in body["notices"])
        baseline_client = TestClient(create_app())
        baseline = baseline_client.post(
            "/api/v1/recommend/diet", json={"tags": ["高尿酸"], "meal": "lunch"}
        ).json()
        assert body == baseline


@pytest.fixture(scope="module")
def index_html():
    client = TestClient(create_app())
    response = client.get("/")
    assert response.status_code == 200
    return response.text


class TestSelfcheckUI:
    """6.7 前端便签锚点（GET / 源码断言）。"""

    def test_entry_button_present(self, index_html):
        assert "📝 情绪自查便签" in index_html
        assert 'id="btn-selfcheck"' in index_html

    def test_modal_anchors_present(self, index_html):
        for anchor in ("selfcheck-modal", "selfcheck-close", "selfcheck-submit", "selfcheck-error", "high-risk-modal", "high-risk-close"):
            assert f'id="{anchor}"' in index_html

    def test_sc3_text_matches_backend(self, index_html):
        assert "发现自己性格有些变化" in index_html

    def test_duration_options_match_enum(self, index_html):
        for option in ("<2 周", "2 周-1 个月", "1-3 个月", "3 个月以上"):
            assert f"'{option}'" in index_html

    def test_c01_hotline_anchors(self, index_html):
        assert "0516-83105638" in index_html
        assert "校内 24 小时心理服务热线" in index_html
        assert 'href="tel:0516-83105638"' in index_html

    def test_frontend_constants_present(self, index_html):
        for name in ("SC_ITEMS", "DURATION_OPTIONS", "HIGH_RISK_SC_ID", "HIGH_RISK_DURATION", "C01_HOTLINE"):
            assert name in index_html
        assert "selfcheck: '/api/v1/mood/selfcheck'" in index_html

    def test_high_risk_warning_script(self, index_html):
        assert "需要立即关注" in index_html
        assert "24 小时随时可以打" in index_html

    def test_function_names_present(self, index_html):
        for fn in ("submitSelfcheck", "renderSelfcheckCard", "buildSelfcheckForm", "collectSelfcheckInput"):
            assert fn in index_html

    def test_no_workspace_memory_json_leaked(self):
        """三纪律之二终验：测试运行不产生工作区真实 memory.json。"""
        if ROOT_MEMORY.exists():
            content = ROOT_MEMORY.read_text(encoding="utf-8")
            assert "stu001" not in content, "测试数据泄漏到工作区 memory.json"

class TestChatHistoryStore:
    """8.1 历史存储锚点（multi_turn_identity tasks 8.1，14 函数）。"""

    def test_new_user_history_empty(self):
        assert get_chat_history("nobody") == []

    def test_save_get_roundtrip_order(self):
        entries = [
            {"role": "user", "content": "午饭吃什么"},
            {"role": "assistant", "content": "推荐如下"},
        ]
        assert save_chat_history("u1", entries) is True
        assert get_chat_history("u1") == entries

    def test_rolling_window_trims_to_latest_twenty(self):
        entries = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"消息{i}"} for i in range(22)]
        save_chat_history("u1", entries)
        stored = get_chat_history("u1")
        assert len(stored) == 20
        assert stored[0]["content"] == "消息2"
        assert stored[-1]["content"] == "消息21"

    def test_user_isolation_histories(self):
        save_chat_history("uA", [{"role": "user", "content": "A 的话题"}])
        save_chat_history("uB", [{"role": "user", "content": "B 的话题"}])
        assert get_chat_history("uA") == [{"role": "user", "content": "A 的话题"}]
        assert get_chat_history("uB") == [{"role": "user", "content": "B 的话题"}]

    def test_persisted_entries_have_exactly_two_keys(self):
        save_chat_history("u1", [{"role": "user", "content": "hi", "extra": "junk"}])
        store = json.loads(memory.MEMORY_FILE_PATH.read_text(encoding="utf-8"))
        for entry in store["u1"]["history"]:
            assert set(entry.keys()) == {"role", "content"}

    def test_missing_and_corrupted_file_return_empty(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setattr(memory, "MEMORY_FILE_PATH", tmp_path / "missing.json")
        assert get_chat_history("u1") == []
        broken = tmp_path / "broken.json"
        broken.write_text("{broken", encoding="utf-8")
        monkeypatch.setattr(memory, "MEMORY_FILE_PATH", broken)
        assert get_chat_history("u1") == []

    def test_get_user_memory_new_user_four_keys(self):
        data = get_user_memory("newbie")
        assert data == {
            "diet_exclusions": [],
            "allergy_history": [],
            "habits": [],
            "history": [],
        }

    def test_save_user_memory_three_view_consistency(self):
        data = {
            "diet_exclusions": ["香菜"],
            "allergy_history": [],
            "habits": ["跑步"],
            "history": [{"role": "user", "content": "hi"}],
        }
        assert save_user_memory("u1", data) is True
        assert get_user_memory("u1") == data
        assert get_memory("u1") == {"diet_exclusions": ["香菜"], "allergy_history": [], "habits": ["跑步"]}
        assert get_chat_history("u1") == [{"role": "user", "content": "hi"}]

    def test_invalid_params_silently_skipped(self):
        before = memory.MEMORY_FILE_PATH.read_text(encoding="utf-8") if memory.MEMORY_FILE_PATH.exists() else ""
        assert save_chat_history("", [{"role": "user", "content": "x"}]) is False
        assert save_chat_history("u1", "not-a-list") is False
        assert save_user_memory("", {}) is False
        after = memory.MEMORY_FILE_PATH.read_text(encoding="utf-8") if memory.MEMORY_FILE_PATH.exists() else ""
        assert before == after

    def test_get_memory_strips_history_key(self):
        save_chat_history("u1", [{"role": "user", "content": "hi"}])
        assert "history" not in get_memory("u1")
        assert set(get_memory("u1").keys()) == {"diet_exclusions", "allergy_history", "habits"}

    def test_add_memory_preserves_history(self):
        save_chat_history("u1", [{"role": "user", "content": "话题"}])
        add_memory("u1", "diet_exclusions", "香菜")
        assert get_chat_history("u1") == [{"role": "user", "content": "话题"}]
        assert get_memory("u1")["diet_exclusions"] == ["香菜"]

    def test_five_rounds_fully_persisted_within_window(self):
        for i in range(5):
            entries = [
                {"role": "user", "content": f"第{i}轮问"},
                {"role": "assistant", "content": f"第{i}轮答"},
            ]
            save_chat_history("u1", get_chat_history("u1") + entries)
        stored = get_chat_history("u1")
        assert len(stored) == 10
        assert stored[0]["content"] == "第0轮问"
        assert stored[-1]["content"] == "第4轮答"

    def test_reread_from_file_after_write(self, tmp_path, monkeypatch):
        """重启模拟基线：写入后重新走模块函数读同一文件。"""
        target = tmp_path / "restart.json"
        monkeypatch.setattr(memory, "MEMORY_FILE_PATH", target)
        save_chat_history("u1", [{"role": "user", "content": "重启前"}])
        monkeypatch.setattr(memory, "MEMORY_FILE_PATH", target)
        assert get_chat_history("u1") == [{"role": "user", "content": "重启前"}]

    def test_history_entries_defense_filtered(self):
        save_chat_history(
            "u1",
            [
                {"role": "user", "content": "合法"},
                {"role": "system", "content": "非法角色"},
                {"content": "缺 role"},
                {"role": "user", "content": ""},
                {"role": "user", "content": "合法2", "extra": "剥"},
                "not-a-dict",
            ],
        )
        stored = get_chat_history("u1")
        assert stored == [
            {"role": "user", "content": "合法"},
            {"role": "user", "content": "合法2"},
        ]


class TestLLMHistoryInjection:
    """8.2 LLM 注入形态锚点（8 函数）。"""

    def _capture_client(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

        return LLMClient(transport=httpx.MockTransport(handler)), captured

    def test_full_messages_shape_with_history(self):
        history = [
            {"role": "user", "content": "第一轮问"},
            {"role": "assistant", "content": "第一轮答"},
            {"role": "user", "content": "第二轮问"},
            {"role": "assistant", "content": "第二轮答"},
        ]
        client, captured = self._capture_client()
        client.chat_with_llm("当前问题", "忌口：香菜", history)
        msgs = captured["body"]["messages"]
        assert len(msgs) == 6
        assert msgs[0]["role"] == "system"
        assert msgs[-1] == {"role": "user", "content": "当前问题"}
        assert [m["role"] for m in msgs] == [
            "system", "user", "assistant", "user", "assistant", "user",
        ]
        assert [m["content"] for m in msgs[1:-1]] == ["第一轮问", "第一轮答", "第二轮问", "第二轮答"]
        assert "忌口：香菜" in msgs[0]["content"]

    def test_none_history_degrades_to_two_messages(self):
        client, captured = self._capture_client()
        client.chat_with_llm("问题", "忌口：香菜", None)
        msgs = captured["body"]["messages"]
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1] == {"role": "user", "content": "问题"}

    def test_empty_list_history_degrades_to_two_messages(self):
        client, captured = self._capture_client()
        client.chat_with_llm("问题", "", [])
        assert len(captured["body"]["messages"]) == 2

    def test_history_time_order_ascending(self):
        history = [
            {"role": "user", "content": "早"},
            {"role": "assistant", "content": "早答"},
            {"role": "user", "content": "晚"},
            {"role": "assistant", "content": "晚答"},
        ]
        client, captured = self._capture_client()
        client.chat_with_llm("现在", "", history)
        contents = [m["content"] for m in captured["body"]["messages"][1:-1]]
        assert contents == ["早", "早答", "晚", "晚答"]

    def test_request_body_no_user_id_leak(self):
        history = [{"role": "user", "content": "历史"}, {"role": "assistant", "content": "答"}]
        client, captured = self._capture_client()
        client.chat_with_llm("当前", "忌口：香菜", history)
        assert "user_id" not in json.dumps(captured["body"])

    def test_oversized_history_truncated_to_latest_six(self):
        history = [{"role": "user", "content": f"消息{i}"} for i in range(8)]
        client, captured = self._capture_client()
        client.chat_with_llm("当前", "", history)
        msgs = captured["body"]["messages"]
        assert len(msgs) == 8
        contents = [m["content"] for m in msgs[1:-1]]
        assert contents == [f"消息{i}" for i in range(2, 8)]

    def test_illegal_entries_filtered(self):
        history = [
            {"role": "user", "content": "合法"},
            {"role": "system", "content": "非法"},
            {"role": "user", "content": 123},
            {"role": "assistant", "content": "合法答", "secret": "剥"},
        ]
        client, captured = self._capture_client()
        client.chat_with_llm("当前", "", history)
        msgs = captured["body"]["messages"]
        assert msgs[1] == {"role": "user", "content": "合法"}
        assert msgs[2] == {"role": "assistant", "content": "合法答"}
        assert len(msgs) == 4

    def test_router_p4_passes_history_third_param(self):
        class ThreeParamCapture:
            def __init__(self):
                self.call_count = 0
                self.history_received = "unset"

            def is_configured(self):
                return True

            def chat_with_llm(self, text, memory_context, history_messages=None):
                self.call_count += 1
                self.history_received = history_messages
                return "兜底回复"

        save_chat_history("u9", [
            {"role": "user", "content": "旧话题"},
            {"role": "assistant", "content": "旧回答"},
        ])
        router = create_default_agent_router()
        stub = ThreeParamCapture()
        router._llm_client = stub
        response = router.route("u9", "徐州有什么好玩的？")
        assert response.reply_text.startswith("兜底回复")
        assert stub.call_count == 1
        assert stub.history_received == [
            {"role": "user", "content": "旧话题"},
            {"role": "assistant", "content": "旧回答"},
        ]

class TestHistoryWindowDecouple:
    """6.4 窗口/注入解耦锚点（quick_dock_history tasks 6.4，6 函数）。"""

    def test_storage_window_constant_is_ten(self):
        assert memory.CHAT_HISTORY_MAX_TURNS == 10

    def test_inject_limit_constant_is_six(self):
        import app.utils.llm as llm_module

        assert llm_module.LLM_HISTORY_INJECT_LIMIT == 6

    def test_llm_module_no_memory_window_import(self):
        from pathlib import Path

        source = Path("app/utils/llm.py").read_text(encoding="utf-8")
        assert "from app.data.memory import CHAT_HISTORY_MAX_TURNS" not in source
        assert "LLM_HISTORY_INJECT_LIMIT" in source

    def test_inject_limit_independent_of_storage_window(self, monkeypatch):
        monkeypatch.setattr(memory, "CHAT_HISTORY_MAX_TURNS", 50)
        history = [{"role": "user", "content": f"消息{i}"} for i in range(8)]
        msgs = build_llm_messages("当前", "", history)
        assert len(msgs) == 8

    def test_storage_twenty_inject_latest_six(self):
        history = [{"role": "user", "content": f"消息{i}"} for i in range(20)]
        save_chat_history("u-dec", history)
        stored = get_chat_history("u-dec")
        assert len(stored) == 20
        msgs = build_llm_messages("当前", "", stored)
        assert len(msgs) == 8
        injected_contents = [m["content"] for m in msgs[1:-1]]
        assert injected_contents == [f"消息{i}" for i in range(14, 20)]

    def test_ten_entries_within_window_fully_stored(self):
        history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"条目{i}"}
            for i in range(10)
        ]
        save_chat_history("u-dec2", history)
        assert get_chat_history("u-dec2") == history

class TestMemoryAllergyNegation:
    """llm_hardening_fitness 组 4 · 过敏项否定排除（独立成类）。"""

    def test_allergy_negation_excluded(self):
        assert extract_memory_preferences("我对海鲜不过敏") == []
        assert extract_memory_preferences("我不过敏") == []

    def test_allergy_positive_kept(self):
        assert extract_memory_preferences("我对海鲜过敏") == [("allergy_history", "海鲜")]

    def test_diet_exclusion_semantics_kept(self):
        assert extract_memory_preferences("我不吃香菜") == [("diet_exclusions", "香菜")]