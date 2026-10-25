"""M4 危机响应链路：项目的"灵魂"模块（spec 5.5）。

固定顺序不变式（设计决策 D7，由 CrisisResponse 字段声明顺序锁定）：
开场话术 → C01→C02→C03→C04→C06→引导语 六项资源优先链 → 收尾话术
→ 抑制标记 → 空推荐结构 → 免责声明。三条安全不变式（S01 硬触发
不可绕过、M4 抑制不可绕过、话术与电话逐字保真）优先级高于一切
其他业务规则；不自行复核 S01（信任上游判定，决策 D8）。
"""

import logging
from typing import Sequence

from app.data.mood_loader import MoodCatalog, create_default_mood_catalog
from app.rules.diet import DietRecommendEngine, create_default_diet_engine
from app.rules.exercise import ExerciseRecommendEngine, create_default_exercise_engine
from app.rules.mood_schemas import (
    CrisisReferral,
    CrisisResponse,
    MatchedSignal,

)

audit_logger = logging.getLogger("coremate.audit")

CRISIS_OPENING = "你现在一个人吗？安全吗？"
CRISIS_CLOSING = "你不用一个人扛。刚才说的这些，听起来已经很难受了。"
GUIDE_TEXT = "如果愿意，可以让你身边任何一个人知道这件事"

CRISIS_RESOURCE_CHAIN: tuple[str, ...] = ("C01", "C02", "C03", "C04", "C06")


class CrisisResponder:
    """M4 危机响应器：停健康建议 → 固定话术 → 资源优先链 → 免责声明置尾。"""

    def __init__(
        self,
        catalog: MoodCatalog,
        diet_engine: DietRecommendEngine,
        exercise_engine: ExerciseRecommendEngine,
    ) -> None:
        self._catalog = catalog
        self._diet_engine = diet_engine
        self._exercise_engine = exercise_engine

    def respond(self, matched_signals: Sequence[MatchedSignal]) -> CrisisResponse:
        """组装 M4 危机响应（每次调用完整执行，同一用户短时多次命中不去重）。"""
        diet_result = self._diet_engine.recommend_suppressed()
        exercise_result = self._exercise_engine.recommend_suppressed()

        referrals = self._assemble_referrals()
        guide_resource = self._find_resource("MR30")
        if guide_resource is not None:
            referrals.append(
                CrisisReferral(
                    res_id=None,
                    name=guide_resource.name or "可以联系身边的人",
                    contact=GUIDE_TEXT,
                    hours=guide_resource.where,
                )
            )
        else:
            audit_logger.warning("引导语资源 MR30 缺失，优先链第六项降级省略")

        response = CrisisResponse(
            opening=CRISIS_OPENING,
            referrals=referrals,
            closing=CRISIS_CLOSING,
            suppression=True,
            diet_recommendations=diet_result,
            exercise_recommendations=exercise_result,
        )
        self._audit(matched_signals, response)
        return response

    def _assemble_referrals(self) -> list[CrisisReferral]:
        """按 MR25-MR30 priority 顺序组装 C 条目资源链；缺失跳过 + 高优先级告警。"""
        chain_resources = {
            record.ref.object_id: record
            for record in self._catalog.mood_resources
            if record.mood_level == "M4" and record.res_type == "referral" and record.ref is not None
        }
        referrals: list[CrisisReferral] = []
        for campus_id in CRISIS_RESOURCE_CHAIN:
            mr_record = chain_resources.get(campus_id)
            campus_record = self._catalog.get_campus(campus_id)
            if mr_record is None or campus_record is None:
                audit_logger.warning("危机资源链条目缺失（%s），跳过但链路不中断", campus_id)
                continue
            referrals.append(
                CrisisReferral(
                    res_id=campus_record.res_id,
                    name=campus_record.name,
                    contact=campus_record.phone or campus_record.location,
                    hours=campus_record.hours,
                    note=campus_record.note,
                )
            )
        return referrals

    def _find_resource(self, res_id: str):
        for record in self._catalog.mood_resources:
            if record.res_id == res_id:
                return record
        return None

    def _audit(self, matched_signals: Sequence[MatchedSignal], response: CrisisResponse) -> None:
        """危机审计日志：只记结构化编号（信号编号 + 资源编号），不落原文与关键词（决策 D10）。"""
        signal_ids = ",".join(item.signal_id for item in matched_signals) or "无"
        resource_ids = ",".join(item.res_id or "GUIDE" for item in response.referrals)
        audit_logger.warning(
            "M4 危机响应触发 | 命中信号=%s | 输出资源=%s | 健康建议已抑制",
            signal_ids,
            resource_ids,
        )



def create_default_crisis_responder() -> CrisisResponder:
    """装配默认危机响应器：注入共享心理数据目录与两个推荐引擎。"""
    return CrisisResponder(
        catalog=create_default_mood_catalog(),
        diet_engine=create_default_diet_engine(),
        exercise_engine=create_default_exercise_engine(),
    )