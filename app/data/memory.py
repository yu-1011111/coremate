"""长期记忆模块：本地 JSON 文件存储用户偏好（memory_llm_ui spec 5.1/5.2）。

纯函数式模块（无实例状态）：add_memory / get_memory 读写契约 +
extract_memory_preferences 抽取纯函数 + build_memory_context 消费辅助。
MEMORY_FILE_PATH 为模块级可 monkeypatch 路径常量（三级上溯对齐
mood_loader.py 先例）；编排层每次经模块函数间接引用（禁止缓存）。
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger("coremate.audit")

MEMORY_FILE_PATH = Path(__file__).resolve().parent.parent.parent / "memory.json"

MEMORY_KEYS: tuple[str, ...] = ("diet_exclusions", "allergy_history", "habits")
MEMORY_KEY_DISPLAY: dict[str, str] = {
    "diet_exclusions": "忌口",
    "allergy_history": "过敏史",
    "habits": "习惯",
}

CHAT_HISTORY_MAX_TURNS = 10
CHAT_HISTORY_KEY = "history"
_HISTORY_ENTRY_ROLES = frozenset({"user", "assistant"})

_MEMORY_REJECT_PATTERN = r"[^，。！？,!?]"

EXTRACTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("diet_exclusions", re.compile(r"我不(?:吃|喝)(?P<item>[^，。！？,!?]{1,12})")),
    ("allergy_history", re.compile(r"我对(?P<item>[^，。！？,!?]{1,12})过敏")),
    ("allergy_history", re.compile(r"我(?P<item>[^，。！？,!?]{1,8})过敏")),
    ("habits", re.compile(r"我(?:习惯|喜欢)(?P<item>[^，。！？,!?]{1,20})")),
)

ALLERGY_EXPANSION_MAP: dict[str, tuple[str, ...]] = {
    "乳糖不耐": ("乳", "奶", "牛奶", "酸奶", "奶油"),
    "海鲜": ("海鲜", "虾", "蟹", "鱼", "贝"),
    "花生": ("花生",),
}

MEMORY_EXTRACTION_DENYLIST: frozenset[str] = frozenset({"药"})

MEMORY_EMPTY: dict[str, list[str]] = {
    "diet_exclusions": [],
    "allergy_history": [],
    "habits": [],
}


def _read_store() -> dict:
    """读取整个记忆库：文件缺失按空库；损坏返回空库并告警。"""
    if not MEMORY_FILE_PATH.is_file():
        return {}
    try:
        payload = json.loads(MEMORY_FILE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("记忆文件读取异常（类型=%s），按空库处理", type(exc).__name__)
        return {}
    if not isinstance(payload, dict):
        logger.warning("记忆文件结构异常，按空库处理")
        return {}
    return payload


def _write_store(store: dict) -> bool:
    """整文件读-改-写持久化：IO 异常内部捕获 + 告警 + 返回 False。"""
    try:
        MEMORY_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        MEMORY_FILE_PATH.write_text(
            json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return True
    except OSError as exc:
        logger.warning("记忆文件写入异常（类型=%s），本次写入丢弃", type(exc).__name__)
        return False


def _normalize_user(store: dict, user_id: str) -> dict[str, list[str]]:
    """取用户记忆字典并补齐三类键（消费方免于判空）。"""
    user_data = store.get(user_id)
    if not isinstance(user_data, dict):
        user_data = {}
    for key in MEMORY_KEYS:
        if not isinstance(user_data.get(key), list):
            user_data[key] = []
    return user_data


def add_memory(user_id: str, key: str, value: str) -> bool:
    """写入一条记忆并持久化：返回 True 新写入 / False 幂等去重跳过。

    key 不在三类记忆键内抛 ValueError 且零文件写入；写入异常静默容错
    返回 False（不打断路由主流程）。
    """
    if key not in MEMORY_KEYS:
        raise ValueError(f"非法记忆类别: {key}")
    if not user_id or not isinstance(value, str) or not value.strip():
        logger.warning("记忆写入参数无效，跳过 | 用户标识是否为空=%s", not bool(user_id))
        return False
    value = value.strip()

    store = _read_store()
    user_data = _normalize_user(store, user_id)
    if value in user_data[key]:
        return False
    user_data[key].append(value)
    store[user_id] = user_data
    return _write_store(store)


def get_memory(user_id: str) -> dict[str, list[str]]:
    """读取用户记忆字典：三类键齐备，缺失/首次出现/损坏均返回空结构。

    白名单式提取（裁决 D2）：仅返回 MEMORY_KEYS 三键——写入历史后
    本函数仍恰返回三类键结构，既有消费方（编排层记忆过滤、
    memory_context 组装）零感知。
    """
    store = _read_store()
    user_data = _normalize_user(store, user_id)
    return {key: user_data.get(key) or [] for key in MEMORY_KEYS}


def _normalize_history_entries(entries) -> list[dict]:
    """历史条目防御过滤纯函数（裁决 D3）：仅保留合法条目并重建两键字典。

    合法条目 = dict 且 role ∈ {user, assistant} 且 content 为非空 str；
    任何附加键被剥离（出境最小化第一道闸）；写入侧与读取侧共用。
    """
    if not isinstance(entries, list):
        return []
    normalized: list[dict] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        role = entry.get("role")
        content = entry.get("content")
        if role not in _HISTORY_ENTRY_ROLES:
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        normalized.append({"role": role, "content": content})
    return normalized


def _history_window_limit() -> int:
    """派生条数上限：轮数 × 2（调整窗口不得改动写入与注入逻辑）。"""
    return CHAT_HISTORY_MAX_TURNS * 2


def get_chat_history(user_id: str) -> list[dict]:
    """读取用户对话历史：无历史（缺失/损坏/首次/存量无键）恒返回 []。

    零副作用零写盘；MEMORY_FILE_PATH 经模块全局名引用。
    """
    store = _read_store()
    user_data = store.get(user_id) if isinstance(store.get(user_id), dict) else {}
    return _normalize_history_entries(user_data.get(CHAT_HISTORY_KEY))


def save_chat_history(user_id: str, history_list) -> bool:
    """写入用户对话历史（防御过滤 → 滚动窗口截断 → 整文件读-改-写）。

    截断在过滤之后执行（保留最新 20 条淘汰最早）；空 user_id / 非列表
    静默跳过返回 False；写入成功时三类偏好键原样保留。
    """
    if not user_id or not isinstance(history_list, list):
        logger.warning("历史写入参数无效，跳过 | 用户标识是否为空=%s", not bool(user_id))
        return False
    entries = _normalize_history_entries(history_list)[-_history_window_limit():]

    store = _read_store()
    user_data = _normalize_user(store, user_id)
    user_data[CHAT_HISTORY_KEY] = entries
    store[user_id] = user_data
    return _write_store(store)


def get_user_memory(user_id: str) -> dict:
    """读取用户段（四键齐备）：三类偏好键 + history（防御过滤结果）。"""
    store = _read_store()
    user_data = _normalize_user(store, user_id)
    user_data[CHAT_HISTORY_KEY] = _normalize_history_entries(user_data.get(CHAT_HISTORY_KEY))
    return user_data


def save_user_memory(user_id: str, data) -> bool:
    """整体覆写用户段并持久化（裁决 D7：原样覆写，不做写入侧键补齐）。"""
    if not user_id or not isinstance(data, dict):
        logger.warning("用户段写入参数无效，跳过 | 用户标识是否为空=%s", not bool(user_id))
        return False
    store = _read_store()
    store[user_id] = data
    return _write_store(store)


def extract_memory_preferences(text: str) -> list[tuple[str, str]]:
    """记忆自动抽取纯函数：按冻结正则扫描文本返回 (key, value) 命中对。

    过敏双模式区间屏蔽：长模式（我对XX过敏）命中区间记入已消费集合，
    短模式（我XX过敏）命中区间与之重叠则丢弃——防止"我对海鲜过敏"
    被短模式重复捕获为"对海鲜"。排除名单命中项丢弃（防"我不吃药"污染
    忌口）。零副作用零写入。
    """
    results: list[tuple[str, str]] = []
    consumed_spans: list[tuple[int, int]] = []
    for key, pattern in EXTRACTION_PATTERNS:
        for match in pattern.finditer(text):
            item = (match.group("item") or "").strip()
            if not item:
                continue
            if item in MEMORY_EXTRACTION_DENYLIST:
                continue
            if key == "allergy_history" and any(char in item for char in ("不", "没", "无")):
                continue
            start, end = match.span("item")
            if any(start < e and end > s for s, e in consumed_spans):
                continue
            consumed_spans.append((start, end))
            results.append((key, item))
    return results


def build_memory_context(memory: dict[str, list[str]]) -> str:
    """组装偏好摘要文本："忌口：香菜、牛奶；过敏史：海鲜；习惯：每天跑步"。

    仅列非空类别；三类全空返回空字符串。
    """
    sections: list[str] = []
    for key in MEMORY_KEYS:
        items = memory.get(key) or []
        if not items:
            continue
        sections.append(f"{MEMORY_KEY_DISPLAY[key]}：{'、'.join(items)}")
    return "；".join(sections)


def expand_allergy_terms(allergy_items: list[str]) -> tuple[str, ...]:
    """过敏原展开：经 ALLERGY_EXPANSION_MAP 展开为匹配词集（含原词）。"""
    terms: list[str] = []
    for item in allergy_items:
        terms.append(item)
        terms.extend(ALLERGY_EXPANSION_MAP.get(item, ()))
    return tuple(terms)