"""知识库优先约束验收测试（kb_grounded_prompt 任务组 4，A1-A11）。

独立测试文件，不修改 tests/test_memory.py（红线保护）；复用
MockTransport 捕获请求体与协议替身手法；tmp_path 隔离记忆路径。
"""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

import app.data.memory as memory
from app.main import create_app
from app.rules.agent_router import create_default_agent_router
from app.utils.llm import (
    LLM_DISCLAIMER_SUFFIX,
    LLM_FALLBACK_SCRIPT,
    LLM_HISTORY_INJECT_LIMIT,
    LLM_KB_CONTEXT_SECTION_TEMPLATE,
    LLM_KB_GROUNDING_INSTRUCTION,
    LLM_KB_SCOPE_NOTE,
    LLM_REPLY_MAX_CHARS,
    LLMClient,
    LLMClientError,
    build_system_prompt,
)
from app.data.knowledge_digest import (
    KB_DIGEST_CAMPUS_FIRST_AID,
    KB_DIGEST_CHRONIC_CARE,
    KB_DIGEST_DIETARY_GUIDE,
    KB_DIGEST_FITNESS_TIPS,
    KB_DIGEST_MEDICAL_CRITERIA,
    KB_DIGEST_PSYCH_INTERVENTION,
    KB_DIGEST_TEXT,
)

KB_MARKER = "【知识库权威上下文】"


@pytest.fixture(autouse=True)
def isolated_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "MEMORY_FILE_PATH", tmp_path / "memory.json")
    yield


def _capture_client():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok 回答"}}]})

    return LLMClient(transport=httpx.MockTransport(handler)), captured


class _LLMStub:
    """协议替身：可编程成功/失败/空回复。"""

    def __init__(self, reply=None, raises=False):
        self._reply = reply
        self._raises = raises
        self.call_count = 0

    def is_configured(self) -> bool:
        return True

    def chat_with_llm(self, text, memory_context, history_messages=None):
        self.call_count += 1
        if self._raises:
            raise LLMClientError("stub failure")
        return self._reply


class TestKbGroundingAnchors:
    """A1-A11 共 11 个锚点。"""

    # A1 指令段逐字锚点
    def test_a1_instruction_in_system_with_three_anchors(self):
        system = build_system_prompt("忌口：香菜")
        assert LLM_KB_GROUNDING_INSTRUCTION in system
        assert "每天食盐不超过5g" in system
        assert "严禁回答6g" in system
        assert "该问题暂未收录" in system

    # A2 独立指令常量逐字锚点（llm_hardening_fitness D-A1 冻结版）
    def test_a2_instruction_constant_frozen(self):
        assert LLM_KB_GROUNDING_INSTRUCTION == (
            "你是芯动管家 CoreMate。在回答以下 5 类问题时，你必须严格检索并提供内部知识库"
            "（慢性病指南、膳食指南、急救手册、心理干预机制、医学准则）中的原话或核心数据，"
            "绝对禁止使用你自己的通用常识作答："
            "1. 【数值与原则】涉及每日摄入量、标准指标（如：每天食盐不超过5g、严禁回答6g、饮水男1700ml、女1500ml）。"
            "2. 【急救与突发不适】涉及晕倒、喝多、呼吸急促、崴脚等（必须给出急救步骤）。"
            "3. 【慢性病生活保养】涉及尿酸高、便秘、幽门螺杆菌的日常注意事项。"
            "4. 【运动误区与技巧】涉及发烧能不能运动、跑步怎么呼吸、考前怎么减量。"
            "5. 【心理调节办法】涉及压力大、孤独、失眠（必须提取具体的干预动作，如呼吸法、宣泄法）。"
            "如果你的内部知识库中没有相关数据，请直接回答'该问题暂未收录专业知识，建议咨询医生'，"
            "严禁使用你自带的通用常识进行捏造。"
        )

    # A3 摘要强锚点
    def test_a3_digest_strong_anchor(self):
        assert "每天摄入食盐不超过 5g" in KB_DIGEST_TEXT
        assert "6g" not in KB_DIGEST_TEXT
        assert "6 克" not in KB_DIGEST_TEXT
        assert "6克" not in KB_DIGEST_TEXT

    # A4 摘要分节锚点
    def test_a4_digest_sections_present(self):
        sections = (
            KB_DIGEST_DIETARY_GUIDE,
            KB_DIGEST_MEDICAL_CRITERIA,
            KB_DIGEST_CHRONIC_CARE,
            KB_DIGEST_CAMPUS_FIRST_AID,
        )
        for section in sections:
            assert section and section.strip()
            assert section.splitlines()[0] in KB_DIGEST_TEXT

    # A5 注入形态锚点
    def test_a5_injection_shape_no_history(self):
        client, captured = _capture_client()
        client.chat_with_llm("为什么要少吃盐", "", None)
        msgs = captured["body"]["messages"]
        assert len(msgs) == 2
        assert KB_MARKER in msgs[0]["content"]
        assert "每天摄入食盐不超过 5g" in msgs[0]["content"]

    # A6 偏好共存 / 无偏好降级锚点
    def test_a6_preference_coexistence_and_empty_degrade(self):
        with_pref = build_system_prompt("忌口：香菜")
        assert "个人偏好" in with_pref and KB_MARKER in with_pref
        empty_pref = build_system_prompt("")
        assert KB_MARKER in empty_pref
        assert "个人偏好" not in empty_pref

    # A7 适用范围限定锚点
    def test_a7_scope_note_limits_four_domains(self):
        assert "医学准则" in LLM_KB_SCOPE_NOTE
        assert "膳食指南" in LLM_KB_SCOPE_NOTE
        assert "急救方法" in LLM_KB_SCOPE_NOTE
        assert "慢性病注意事项" in LLM_KB_SCOPE_NOTE
        assert "暂时未收录" not in LLM_KB_SCOPE_NOTE
        assert "个人偏好" not in LLM_KB_SCOPE_NOTE
        assert "个人偏好" not in LLM_KB_CONTEXT_SECTION_TEMPLATE

    # A8 摘要为空降级锚点
    def test_a8_empty_digest_degrades_without_exception(self, monkeypatch):
        import app.data.knowledge_digest as digest_module

        monkeypatch.setattr(digest_module, "KB_DIGEST_TEXT", "")
        system = build_system_prompt("忌口：香菜")
        assert KB_MARKER not in system
        assert "芯动管家" in system

    # A9 消息条数不变锚点
    def test_a9_message_count_unchanged(self):
        client, captured = _capture_client()
        history = [{"role": "user", "content": f"消息{i}"} for i in range(8)]
        client.chat_with_llm("当前", "忌口：香菜", history)
        msgs = captured["body"]["messages"]
        assert len(msgs) == 2 + LLM_HISTORY_INJECT_LIMIT
        assert [m["role"] for m in msgs[:3]] == ["system", "user", "user"]
        assert "忌口：香菜" in msgs[0]["content"]
        assert msgs[-1] == {"role": "user", "content": "当前"}

    # A10 降级链不变锚点
    def test_a10_fallback_chain_unchanged(self):
        router = create_default_agent_router()
        fail_stub = _LLMStub(raises=True)
        router._llm_client = fail_stub
        failed = router.route("u1", "量子力学是什么")
        assert failed.reply_text == LLM_FALLBACK_SCRIPT

        router2 = create_default_agent_router()
        router2._llm_client = _LLMStub(reply="字" * 1500)
        success = router2.route("u1", "讲个长故事")
        assert success.reply_text.endswith(LLM_DISCLAIMER_SUFFIX)
        assert len(success.reply_text) - len("\n" + LLM_DISCLAIMER_SUFFIX) == LLM_REPLY_MAX_CHARS

        router3 = create_default_agent_router()
        router3._llm_client = _LLMStub(reply="   ")
        empty = router3.route("u1", "你好呀")
        assert empty.reply_text.startswith(LLM_FALLBACK_SCRIPT)

    # A11 P1-P3 零影响锚点
    @pytest.mark.parametrize(
        "text",
        ["我想死", "我想去网吧", "难受，怎么办", "尿酸高，午饭吃什么？", "我想吃红烧肉"],
    )
    def test_a11_p1_p3_never_call_llm(self, text):
        router = create_default_agent_router()
        stub = _LLMStub(reply="不应调用")
        router._llm_client = stub
        response = router.route("u1", text)
        assert stub.call_count == 0
        assert response.route is not None

    # 端到端：Ask "why less salt" 的 system 注入（API 级）
    def test_api_chat_carries_kb_context_to_llm(self):
        application = create_app()
        captured = {}

        class CapturingLLM:
            def is_configured(self):
                return True

            def chat_with_llm(self, text, memory_context, history_messages=None):
                captured["system"] = build_system_prompt(memory_context)
                return "每天摄入食盐不超过 5g。"

        application.state.agent_router._llm_client = CapturingLLM()
        client = TestClient(application)
        body = client.post(
            "/api/v1/chat", json={"user_id": "u-salt", "text": "为什么要少吃盐"}
        ).json()
        assert body["route"] == "SMALL_TALK"
        assert "每天摄入食盐不超过 5g" in body["reply_text"]
        assert KB_MARKER in captured["system"]
        assert "每天摄入食盐不超过 5g" in captured["system"]

class TestKbHardeningFitnessAnchors:
    """llm_hardening_fitness 组 1 · 知识库强约束（独立成类）。"""

    def test_instruction_contains_five_domains_and_anchors(self):
        for anchor in (
            "1. 【数值与原则】",
            "2. 【急救与突发不适】",
            "3. 【慢性病生活保养】",
            "4. 【运动误区与技巧】",
            "5. 【心理调节办法】",
            "盐不超过5g",
            "严禁回答6g",
            "饮水男1700ml",
            "女1500ml",
            "该问题暂未收录专业知识，建议咨询医生",
        ):
            assert anchor in LLM_KB_GROUNDING_INSTRUCTION

    def test_digest_six_sections_first_line_present(self):
        sections = (
            KB_DIGEST_DIETARY_GUIDE,
            KB_DIGEST_MEDICAL_CRITERIA,
            KB_DIGEST_CHRONIC_CARE,
            KB_DIGEST_CAMPUS_FIRST_AID,
            KB_DIGEST_FITNESS_TIPS,
            KB_DIGEST_PSYCH_INTERVENTION,
        )
        for section in sections:
            assert section and section.strip()
            assert section.splitlines()[0] in KB_DIGEST_TEXT

    def test_fitness_digest_anchors(self):
        assert "3-3" in KB_DIGEST_FITNESS_TIPS
        assert "2-2" in KB_DIGEST_FITNESS_TIPS
        assert "考前 3 天减量" in KB_DIGEST_FITNESS_TIPS
        assert "发烧" in KB_DIGEST_FITNESS_TIPS

    def test_psych_digest_anchors(self):
        assert "4-6 呼吸" in KB_DIGEST_PSYCH_INTERVENTION
        assert "宣泄法" in KB_DIGEST_PSYCH_INTERVENTION

    def test_digest_no_forbidden_gram_anchor(self):
        assert "6g" not in KB_DIGEST_TEXT
        assert "6 克" not in KB_DIGEST_TEXT
        assert "6克" not in KB_DIGEST_TEXT

    def test_scope_note_covers_five_domains(self):
        for domain in ("医学准则", "膳食指南", "急救方法", "慢性病注意事项", "心理调节"):
            assert domain in LLM_KB_SCOPE_NOTE