"""智谱 LLM 兜底客户端：glm-4-flash 对话（memory_llm_ui spec 5.3）。

同步短生命周期客户端（设计决策 D1：route() 同步签名约束，对齐
WeatherClient 先例）；凭证与 vision.py 共用 ZHIPU_API_KEY 变量名但
各自独立判定（决策 D6）。失败统一抛 LLMClientError，由编排层降级。
"""

import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger("coremate.data")

# 凭证读取：默认值初始必须为占位值（未配置态开箱，纪律之一）。
# 注意：与 app/utils/vision.py 的同名常量需维护者保持同步（各自独立读取）。
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "YOUR_ZHIPU_KEY_HERE")

ZHIPU_CHAT_COMPLETIONS_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
LLM_MODEL = "glm-4-flash"
LLM_TIMEOUT_SECONDS = 10.0
LLM_REPLY_MAX_CHARS = 1000
LLM_HISTORY_INJECT_LIMIT = 6

LLM_SYSTEM_PROMPT_TEMPLATE = (
    "你是芯动管家 CoreMate，一名贴心的校园健康管理助手。"
    "你只回答与健康生活、饮食运动、校园日常相关的通用问题；"
    "遇到涉及疾病诊断、用药、心理危机的问题，提醒用户咨询校医院或专业机构。"
    "回答保持简洁友好，控制在 200 字以内。\n{kb_section}\n{memory_section}"
)
LLM_MEMORY_SECTION_TEMPLATE = (
    "你已知该用户的个人偏好，回答时请自然地照顾这些偏好：{memory_summary}。"
)
LLM_KB_GROUNDING_INSTRUCTION = (
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
LLM_KB_SCOPE_NOTE = (
    "上述要求仅适用于五类专业问题（医学准则、膳食指南、急救方法、慢性病注意事项、心理调节办法）；"
    "普通闲聊与常识问题请照常友好回答，不得回答'该问题暂未收录'。"
)
LLM_KB_CONTEXT_SECTION_TEMPLATE = (
    "【知识库权威上下文】\n{kb_digest}\n以上为可优先引用的权威数据；引用时保持简洁（200 字以内）。"
)
LLM_KB_SECTION_TEMPLATE = "{kb_instruction}\n{kb_scope_note}\n\n{kb_context_section}"
LLM_DISCLAIMER_SUFFIX = "（本建议仅供日常健康管理参考）"
LLM_FALLBACK_SCRIPT = (
    "抱歉，这个我还在学习中。你可以问我今天该吃什么，或者去操场该做什么运动。"
)

_KEY_PLACEHOLDER = "YOUR_ZHIPU_KEY_HERE"


class LLMClientError(Exception):
    """LLM 对话调用失败（网络/超时/非 2xx/响应结构异常统一形态）。"""


def is_configured() -> bool:
    """单变量判定：Key 非空且非占位值。"""
    return bool(ZHIPU_API_KEY) and ZHIPU_API_KEY != _KEY_PLACEHOLDER


def _build_kb_section() -> str:
    """组装知识库优先段（kb_grounded_prompt 2.6）：指令段 + 适用范围段 + 上下文段。

    摘要为空 → 整段省略并告警；指令缺失/为空 → 按无指令段处理；任何
    异常均降级为空段，不抛异常。注入文本一律作为 format 参数值传入。
    """
    try:
        from app.data.knowledge_digest import KB_DIGEST_TEXT
    except Exception as exc:
        logger.warning("知识库摘要导入异常（类型=%s），按无上下文处理", type(exc).__name__)
        return ""

    digest = (KB_DIGEST_TEXT or "").strip()
    if not digest:
        logger.warning("知识库摘要为空，知识库上下文段整段省略")
        return ""

    instruction = (LLM_KB_GROUNDING_INSTRUCTION or "").strip()
    if not instruction:
        logger.warning("知识库优先指令缺失，按无指令段处理")
    scope_note = (LLM_KB_SCOPE_NOTE or "").strip()
    context_section = LLM_KB_CONTEXT_SECTION_TEMPLATE.format(kb_digest=digest)
    return LLM_KB_SECTION_TEMPLATE.format(
        kb_instruction=instruction,
        kb_scope_note=scope_note,
        kb_context_section=context_section,
    ).strip()


def build_system_prompt(memory_context: str) -> str:
    """组装 system 消息：次序固定 persona → 知识库优先段 → 偏好段。

    memory_context 为空时偏好段为空串；知识库上下文段始终注入（仅 P4
    兜底路径消费 system，P1-P3 不经 LLM 天然零影响）。
    """
    kb_section = _build_kb_section()
    if not memory_context:
        return LLM_SYSTEM_PROMPT_TEMPLATE.format(kb_section=kb_section, memory_section="")
    section = LLM_MEMORY_SECTION_TEMPLATE.format(memory_summary=memory_context)
    return LLM_SYSTEM_PROMPT_TEMPLATE.format(kb_section=kb_section, memory_section=section)


def build_llm_messages(text: str, memory_context: str, history_messages=None) -> list:
    """组装完整消息序列：[system(含记忆)] + 历史对(≤6，时间序) + [user(当前)]。

    历史归一（None → []）→ 条目级防御过滤 → 超上限取最新 6 条（上限
    消费独立常量 LLM_HISTORY_INJECT_LIMIT，与存储窗口解耦）；空历史
    退化为既有双消息形态（system + user，与接入前逐字节一致）；产出
    全新字典列表；纯函数不抛异常（输入类型异常按无历史处理）。
    """
    if not isinstance(history_messages, list):
        history_messages = []

    def _valid(entry) -> bool:
        if not isinstance(entry, dict):
            return False
        role = entry.get("role")
        content = entry.get("content")
        return role in ("user", "assistant") and isinstance(content, str) and bool(content.strip())

    entries = [
        {"role": e["role"], "content": e["content"]}
        for e in history_messages
        if _valid(e)
    ][-LLM_HISTORY_INJECT_LIMIT:]
    return (
        [{"role": "system", "content": build_system_prompt(memory_context)}]
        + entries
        + [{"role": "user", "content": text}]
    )


class LLMClient:
    """智谱 glm-4-flash 同步对话客户端（短生命周期，无实例可变状态）。"""

    def __init__(self, transport: Optional[httpx.BaseTransport] = None) -> None:
        self._transport = transport

    @staticmethod
    def is_configured() -> bool:
        """凭证判定（委托模块级判定）。"""
        return is_configured()

    def chat_with_llm(self, text: str, memory_context: str, history_messages=None) -> str:
        """单轮对话：system（含偏好注入）+ 历史对（≤6）+ user（文本）→ 回复。

        任何失败（网络/超时/非 2xx/结构缺失/content 非文本）统一抛
        LLMClientError；告警仅含失败类型，凭证零落盘。
        """
        payload = {
            "model": LLM_MODEL,
            "messages": build_llm_messages(text, memory_context, history_messages),
        }
        try:
            with httpx.Client(
                headers={"Authorization": f"Bearer {ZHIPU_API_KEY}"},
                timeout=LLM_TIMEOUT_SECONDS,
                transport=self._transport,
            ) as client:
                response = client.post(ZHIPU_CHAT_COMPLETIONS_URL, json=payload)
        except Exception as exc:
            logger.warning("LLM 兜底调用异常（类型=%s）", type(exc).__name__)
            raise LLMClientError("LLM 调用失败") from exc

        if response.status_code != 200:
            logger.warning("LLM 兜底非 2xx 响应: %s", response.status_code)
            raise LLMClientError("LLM 服务响应异常")
        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            logger.warning("LLM 兜底响应结构异常")
            raise LLMClientError("LLM 响应结构异常") from exc
        if not isinstance(content, str):
            raise LLMClientError("LLM 回复非文本")
        return content.strip()