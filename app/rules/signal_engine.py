"""情绪信号识别引擎：对话文本 + 行为事件 + 自报事件 → 档位判定（spec 5.3）。

S01（自杀/自伤）命中即硬触发 M4 为全模块最高优先级不变式（决策 D5
前置短路）：由控制流结构保证不可绕过，禁止被任何过滤、降级或异常
处理路径屏蔽。weight 累加与最高档兜底的输出域显式限制 M1-M3，
任何 weight 组合禁止产出 M4。
"""

import logging
from typing import Optional, Sequence

from app.data.mood_loader import MoodCatalog, create_default_mood_catalog
from app.rules.mood_schemas import (
    EMPTY_SIGNAL_INPUT_ERROR,
    MOOD_DATA_UNAVAILABLE_NOTICE,
    MOOD_RANK,
    NO_MOOD_ISSUE,
    WEIGHT_THRESHOLDS,
    BehaviorEvent,
    MatchedSignal,
    MoodResourceItem,
    MoodSignalRecord,
    SignalEmptyInputError,
    SignalResult,
)

audit_logger = logging.getLogger("coremate.audit")

BEHAVIOR_KEYWORD_MAP: dict[str, str] = {
    "深夜活跃": "深夜活跃",
    "低步数": "久坐不动",
    "运动停滞": "运动停滞",
    "饮食压抑": "饮食压抑",
    "夜间游荡": "夜间游荡",
}

MOOD_TEXT_EXTENSION_KEYWORDS: dict[str, str] = {
    "快疯了": "M2",
    "砸东西": "M2",
    "emo": "M1",
    "破防": "M2",
    "摆烂": "M2",
    "心塞": "M2",
    "心情很差": "M1",
}

MOOD_TEXT_EXTENSION_SIGNAL_ID = "EXT-M2"

S01_POSITIVE_NEGATION_PHRASES: tuple[str, ...] = ("不想死", "不想自残", "不会自杀")

S01_EXEMPTION_SIGNAL_ID: str = "S01-EXEMPT"


def _find_all_spans(text: str, keyword: str) -> list[tuple[int, int]]:
    """枚举关键词在文本中的全部出现区间（find 循环推进）。"""
    spans: list[tuple[int, int]] = []
    start = text.find(keyword)
    while start >= 0:
        spans.append((start, start + len(keyword)))
        start = text.find(keyword, start + len(keyword))
    return spans


def _s01_hit_spans(text: str, keywords: Sequence[str]) -> list[tuple[str, int, int]]:
    """S01 全部命中区间枚举：(关键词, start, end)，关键词取目录 S01 原文。"""
    spans: list[tuple[str, int, int]] = []
    for keyword in keywords:
        if not keyword:
            continue
        for start, end in _find_all_spans(text, keyword):
            spans.append((keyword, start, end))
    return spans


def _s01_positive_negation_exempted(
    text: str, s01_hit_spans: list[tuple[str, int, int]]
) -> bool:
    """S01 正向豁免判定（D-S01EX，fail-closed）。

    仅当三枚举短语的命中区间能完全覆盖**每一个** S01 命中区间时豁免；
    任何异常按未豁免处理（返回 False → 维持 M4，极保守）。
    """
    try:
        exempt_spans: list[tuple[int, int]] = []
        for phrase in S01_POSITIVE_NEGATION_PHRASES:
            exempt_spans.extend(_find_all_spans(text, phrase))
        if not exempt_spans:
            return False
        return all(
            any(es <= s and e <= ee for (es, ee) in exempt_spans)
            for (_, s, e) in s01_hit_spans
        )
    except Exception as exc:
        audit_logger.warning("S01 正向豁免判定异常（类型=%s），按未豁免处理", type(exc).__name__)
        return False


class SignalEngine:
    """情绪信号识别引擎：S01 硬触发 → 文本/行为/自报匹配 → 累加与兜底。"""

    def __init__(self, catalog: MoodCatalog) -> None:
        self._catalog = catalog

    def analyze(
        self,
        text: str,
        behavior_events: Sequence[BehaviorEvent] = (),
        self_reports: Sequence[str] = (),
    ) -> SignalResult:
        """执行情绪信号识别：输入对话文本与行为/自报事件。"""
        if not self._catalog.signals:
            result = SignalResult(notices=[MOOD_DATA_UNAVAILABLE_NOTICE])
            self._audit(text, behavior_events, result)
            return result

        if not text and not behavior_events and not self_reports:
            raise SignalEmptyInputError(EMPTY_SIGNAL_INPUT_ERROR)

        notices: list[str] = []
        matched: list[MatchedSignal] = []

        s01 = self._catalog.get_signal("S01")
        s01_hit = s01 is not None and any(keyword and keyword in text for keyword in s01.keywords)
        s01_exempted = False
        if s01_hit and s01 is not None:
            try:
                hit_spans = _s01_hit_spans(text, s01.keywords)
                s01_exempted = _s01_positive_negation_exempted(text, hit_spans)
            except Exception as exc:
                audit_logger.warning(
                    "S01 正向豁免判定异常（类型=%s），按未豁免处理", type(exc).__name__
                )
                s01_exempted = False
        if s01_hit and s01 is not None:
            if s01_exempted:
                matched.append(
                    MatchedSignal(
                        signal_id=S01_EXEMPTION_SIGNAL_ID,
                        hit_keywords=[
                            phrase
                            for phrase in S01_POSITIVE_NEGATION_PHRASES
                            if phrase in text
                        ],
                    )
                )
                audit_logger.info(
                    "S01 正向豁免生效 | 豁免短语命中=%s | 覆盖命中数=%d",
                    ",".join(phrase for phrase in S01_POSITIVE_NEGATION_PHRASES if phrase in text),
                    len(hit_spans),
                )
            else:
                matched.append(
                    MatchedSignal(
                        signal_id=s01.signal_id,
                        hit_keywords=[keyword for keyword in sorted(s01.keywords) if keyword in text],
                    )
                )

        other_signals = [s for s in self._catalog.signals if s.signal_id != "S01"]
        total_weight = 0
        for signal in other_signals:
            hit_keywords = self._match_signal(signal, text, behavior_events, self_reports)
            if hit_keywords:
                matched.append(MatchedSignal(signal_id=signal.signal_id, hit_keywords=hit_keywords))
                total_weight += signal.weight

        extension_hits = [keyword for keyword in sorted(MOOD_TEXT_EXTENSION_KEYWORDS) if keyword and keyword in text]
        if extension_hits:
            matched.append(
                MatchedSignal(signal_id=MOOD_TEXT_EXTENSION_SIGNAL_ID, hit_keywords=extension_hits)
            )

        mood_level = self._determine_mood(s01_hit and not s01_exempted, total_weight, matched)

        mood_resources: list[MoodResourceItem] = []
        if (not s01_hit or s01_exempted) and mood_level in ("M1", "M2", "M3"):
            mood_resources = self._collect_mood_resources(mood_level)

        result = SignalResult(
            matched_signals=matched,
            total_weight=total_weight,
            mood_level=mood_level,
            mood_resources=mood_resources,
            notices=notices,
        )
        self._audit(text, behavior_events, result)
        return result

    def _match_signal(
        self,
        signal: MoodSignalRecord,
        text: str,
        behavior_events: Sequence[BehaviorEvent],
        self_reports: Sequence[str],
    ) -> list[str]:
        """单信号匹配：文本关键词 / 行为事件阈值 / 自报 SC 编号。"""
        hit_keywords: list[str] = [keyword for keyword in sorted(signal.keywords) if keyword and keyword in text]

        if signal.category == "行为":
            for event in behavior_events:
                keyword_label = BEHAVIOR_KEYWORD_MAP.get(event.event_type.value)
                if (
                    keyword_label
                    and keyword_label in signal.keywords
                    and event.consecutive_days >= signal.min_consecutive_days
                ):
                    hit_keywords.append(keyword_label)
                    break

        if signal.category == "自报" and signal.sc_ids:
            for sc_id in self_reports:
                if sc_id in signal.sc_ids:
                    hit_keywords.append(f"自报{sc_id}")
                    break

        return hit_keywords

    def _determine_mood(
        self, s01_hit: bool, total_weight: int, matched: Sequence[MatchedSignal]
    ) -> str:
        """档位判定：S01 硬触发 M4；否则累加映射与 target_mood 最高档取高（限 M1-M3）。"""
        if s01_hit:
            return "M4"

        accumulated = NO_MOOD_ISSUE
        if total_weight > 0:
            for low, high, mood in WEIGHT_THRESHOLDS:
                if low <= total_weight <= high:
                    accumulated = mood
                    break

        highest_target: Optional[str] = None
        for item in matched:
            target_mood = self._resolve_target_mood(item.signal_id, item.hit_keywords)
            if target_mood not in ("M1", "M2", "M3"):
                continue
            if highest_target is None or MOOD_RANK[target_mood] > MOOD_RANK[highest_target]:
                highest_target = target_mood

        candidates = [mood for mood in (accumulated, highest_target) if mood in MOOD_RANK]
        if not candidates:
            return NO_MOOD_ISSUE
        return max(candidates, key=lambda m: MOOD_RANK[m])

    def _resolve_target_mood(
        self, signal_id: str, hit_keywords: Sequence[str] = ()
    ) -> Optional[str]:
        """解析信号目标档位：补充/豁免信号按实际命中词取档，其余查目录。"""
        if signal_id == S01_EXEMPTION_SIGNAL_ID:
            return "M2"
        if signal_id == MOOD_TEXT_EXTENSION_SIGNAL_ID:
            moods = [
                MOOD_TEXT_EXTENSION_KEYWORDS[keyword]
                for keyword in hit_keywords
                if keyword in MOOD_TEXT_EXTENSION_KEYWORDS
            ]
            if not moods:
                return None
            return max(moods, key=lambda mood: MOOD_RANK[mood])
        signal = self._catalog.get_signal(signal_id)
        return signal.target_mood if signal is not None else None

    def _collect_mood_resources(self, mood_level: str) -> list[MoodResourceItem]:
        """档位资源匹配：mood_level 精确一致 + priority 升序，排除 SLEEP（M4 隔离）。"""
        matched = [
            record
            for record in self._catalog.mood_resources
            if record.mood_level == mood_level and record.mood_level != "SLEEP"
        ]
        matched.sort(key=lambda r: (r.priority, r.res_id))
        return [
            MoodResourceItem(
                res_id=record.res_id,
                mood_level=record.mood_level,
                res_type=record.res_type,
                name=record.name,
                minutes=record.minutes,
                where=record.where,
                mechanism=record.mechanism,
                priority=record.priority,
                ref=record.ref,
            )
            for record in matched
        ]

    def _audit(self, text: str, behavior_events: Sequence[BehaviorEvent], result: SignalResult) -> None:
        """审计日志：记录文本长度、事件数与输出档位，不落对话原文（spec 4.3.4）。"""
        audit_logger.info(
            "情绪信号识别完成 | 文本长度=%d | 行为事件数=%d | 命中信号=%s | 总分=%d | 档位=%s",
            len(text),
            len(behavior_events),
            ",".join(item.signal_id for item in result.matched_signals) or "无",
            result.total_weight,
            result.mood_level,
        )


def create_default_signal_engine() -> SignalEngine:
    """装配默认情绪信号识别引擎：注入共享心理数据目录单例。"""
    return SignalEngine(catalog=create_default_mood_catalog())