"""便签自查引擎：勾选 SC1-SC9 + 持续时长 → 情绪档位与应对办法（spec 5.2）。

无状态线程安全、纯本地集合运算；M4 保护规则：自查路径任何组合
禁止输出 M4（封顶 M3），M4 只能由信号引擎 S01 硬触发。
"""

import logging
from typing import Sequence

from app.data.mood_loader import MoodCatalog, create_default_mood_catalog
from app.rules.mood_schemas import (
    KB_MISSING_NOTICE,
    LONG_DURATION_REFERRAL_NOTICE,
    MOOD_DATA_UNAVAILABLE_NOTICE,
    MOOD_RANK,
    NO_VALID_SC_ERROR,
    REF_UNAVAILABLE_NOTICE,
    DurationLevel,
    KbRecommendation,
    MoodResourceItem,
    ReferralInfo,
    SelfcheckInvalidInputError,
    SelfcheckResult,
)

audit_logger = logging.getLogger("coremate.audit")

_DURATION_SHIFT: dict[DurationLevel, int] = {
    DurationLevel.UNDER_2W: 0,
    DurationLevel.W2_TO_1M: 0,
    DurationLevel.M1_TO_3M: 1,
    DurationLevel.OVER_3M: 1,
}

_MOOD_ORDER = ("M1", "M2", "M3")


def _shift_mood(mood: str, steps: int) -> str:
    """档位上调（限 M1-M3 输出域，封顶 M3）。"""
    if mood not in _MOOD_ORDER:
        return mood
    idx = min(_MOOD_ORDER.index(mood) + steps, len(_MOOD_ORDER) - 1)
    return _MOOD_ORDER[idx]


def _rank_of(mood: str) -> int:
    return MOOD_RANK.get(mood, 0)


class SelfcheckEngine:
    """便签自查引擎：勾选校验 → 基础档位 → 时长修正（封顶 M3）→ 手册/转介/资源输出。"""

    def __init__(self, catalog: MoodCatalog) -> None:
        self._catalog = catalog

    def evaluate(self, checked_sc_ids: Sequence[str], duration: DurationLevel) -> SelfcheckResult:
        """执行便签自查判定：输入勾选编号与时长档位。"""
        if not self._catalog.selfcheck_items:
            result = SelfcheckResult(mood_level="M1", notices=[MOOD_DATA_UNAVAILABLE_NOTICE])
            self._audit(checked_sc_ids, result)
            return result

        notices: list[str] = []
        valid_items = []
        unknown_ids: list[str] = []
        for sc_id in checked_sc_ids:
            record = self._catalog.get_selfcheck(sc_id)
            if record is None:
                unknown_ids.append(sc_id)
                continue
            valid_items.append(record)
        if unknown_ids:
            notices.append("存在未识别的自查条目，已被忽略")
        if not valid_items:
            raise SelfcheckInvalidInputError(NO_VALID_SC_ERROR)

        base_mood = max((item.mood_level for item in valid_items), key=_rank_of)
        final_mood = _shift_mood(base_mood, _DURATION_SHIFT[duration])

        kb_recommendations = self._collect_kb_recommendations(valid_items, notices)
        referrals = self._collect_referrals(valid_items)
        if duration is DurationLevel.OVER_3M and not referrals:
            fallback = self._catalog.get_campus("C04")
            if fallback is not None:
                referrals.append(_to_referral_info(fallback))
                notices.append(LONG_DURATION_REFERRAL_NOTICE)
        elif duration is DurationLevel.OVER_3M:
            notices.append(LONG_DURATION_REFERRAL_NOTICE)

        mood_resources = self._collect_mood_resources(final_mood)

        result = SelfcheckResult(
            mood_level=final_mood,
            kb_recommendations=kb_recommendations,
            referrals=referrals,
            mood_resources=mood_resources,
            notices=notices,
        )
        self._audit(checked_sc_ids, result)
        return result

    def _collect_kb_recommendations(
        self, items: Sequence, notices: list[str]
    ) -> list[KbRecommendation]:
        """kb_ids 解析为完整手册条目（含 link 引用）；缺失跳过 + 提示。"""
        recommendations: list[KbRecommendation] = []
        missing = False
        for item in items:
            for kb_id in sorted(item.kb_ids):
                record = self._catalog.get_kb(kb_id)
                if record is None:
                    missing = True
                    continue
                unavailable_links = any(not link.available for link in record.links)
                recommendations.append(
                    KbRecommendation(
                        kb_id=record.kb_id,
                        method=record.method,
                        howto=record.howto,
                        links=list(record.links),
                    )
                )
                if unavailable_links:
                    notices.append(REF_UNAVAILABLE_NOTICE)
        if missing:
            notices.append(KB_MISSING_NOTICE)
        return recommendations

    def _collect_referrals(self, items: Sequence) -> list[ReferralInfo]:
        """has_referral=1 的勾选项 → referral_id 指向的校园求助资源（去重保序）。"""
        seen: set[str] = set()
        referrals: list[ReferralInfo] = []
        for item in items:
            if not item.has_referral:
                continue
            for res_id in sorted(item.referral_ids):
                if res_id in seen:
                    continue
                record = self._catalog.get_campus(res_id)
                if record is None:
                    continue
                seen.add(res_id)
                referrals.append(_to_referral_info(record))
        return referrals

    def _collect_mood_resources(self, mood_level: str) -> list[MoodResourceItem]:
        """档位资源匹配：mood_level 精确一致 + priority 升序，排除 SLEEP。"""
        matched = [
            record
            for record in self._catalog.mood_resources
            if record.mood_level == mood_level and record.mood_level != "SLEEP"
        ]
        matched.sort(key=lambda r: (r.priority, r.res_id))
        return [_to_resource_item(record) for record in matched]

    def _audit(self, checked_sc_ids: Sequence[str], result: SelfcheckResult) -> None:
        """审计日志：仅记录勾选数与输出档位摘要，不落便签原文（spec 4.4.4）。"""
        audit_logger.info(
            "便签自查完成 | 勾选条目数=%d | 输出档位=%s | 手册条目=%d | 转介=%d",
            len(checked_sc_ids),
            result.mood_level,
            len(result.kb_recommendations),
            len(result.referrals),
        )


def _to_referral_info(record) -> ReferralInfo:
    return ReferralInfo(
        res_id=record.res_id,
        name=record.name,
        contact=record.phone or record.location,
        hours=record.hours,
        note=record.note,
    )


def _to_resource_item(record) -> MoodResourceItem:
    return MoodResourceItem(
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


def create_default_selfcheck_engine() -> SelfcheckEngine:
    """装配默认便签自查引擎：注入共享心理数据目录单例。"""
    return SelfcheckEngine(catalog=create_default_mood_catalog())