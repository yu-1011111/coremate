"""体育课选课建议引擎：依据健康标签匹配体育课库（spec 5.5）。

输入仅健康标签，无场景/时长/天气维度；block/recommend/caution 三路
判定 + 安全优先归并（design 决策 D9）。
"""

import logging
from typing import Sequence

from app.data.rec_catalog import RecCatalog, create_default_catalog
from app.rules.rec_schemas import (
    DATA_UNAVAILABLE_NOTICE,
    NO_COURSE_NOTICE,
    UNKNOWN_TAG_NOTICE,
    AvoidedCourse,
    CautionNotice,
    RecommendedCourse,
    SportsCourseRecommendResult,

    extract_reason_for_tag,
)
from app.rules.tag_mapping import map_health_tags

audit_logger = logging.getLogger("coremate.audit")

HEALTHY_COURSE_REASON = "未命中任何禁忌条件，可作为常规选课"


class SportsCourseAdvisor:
    """体育课选课建议引擎：全量课目三路判定（安全优先）。"""

    def __init__(self, catalog: RecCatalog) -> None:
        self._catalog = catalog

    def recommend(self, tags: Sequence[str]) -> SportsCourseRecommendResult:
        """执行选课建议：输入健康标签，输出推荐/避雷课目与谨慎提示。"""
        if not self._catalog.sport_courses:
            result = SportsCourseRecommendResult(notices=[DATA_UNAVAILABLE_NOTICE])
            self._audit(tags, result)
            return result

        mapped = map_health_tags(tags)
        notices: list[str] = []
        if mapped.unknown_tags:
            notices.append(UNKNOWN_TAG_NOTICE)

        crowd = mapped.crowd_tags
        recommended: list[RecommendedCourse] = []
        avoided: list[AvoidedCourse] = []
        caution_notices: list[CautionNotice] = []

        if crowd:
            for course in self._catalog.sport_courses:
                hit_block = course.block_for & crowd
                if hit_block:
                    tag = sorted(hit_block)[0]
                    reason = extract_reason_for_tag(course.block_reason, tag) or f"命中 block_for 标注：{tag}"
                    avoided.append(AvoidedCourse(course_id=course.course_id, name=course.name, reason=reason))
                    continue
                hit_recommend = course.recommend_for & crowd
                if hit_recommend:
                    tag = sorted(hit_recommend)[0]
                    reason = extract_reason_for_tag(course.recommend_reason, tag) or f"适合{tag}人群"
                    recommended.append(
                        RecommendedCourse(
                            course_id=course.course_id,
                            name=course.name,
                            reason=reason,
                            fitness_1000m=course.fitness_1000m,
                        )
                    )
                hit_caution = course.caution_for & crowd
                if hit_caution:
                    tag = sorted(hit_caution)[0]
                    reason = extract_reason_for_tag(course.caution_reason, tag) or f"命中 caution_for 标注：{tag}"
                    caution_notices.append(CautionNotice(kind="course", target=course.name, reason=reason))
            if not recommended:
                notices.append(NO_COURSE_NOTICE)
        else:
            for course in self._catalog.sport_courses:
                recommended.append(
                    RecommendedCourse(
                        course_id=course.course_id,
                        name=course.name,
                        reason=HEALTHY_COURSE_REASON,
                        fitness_1000m=course.fitness_1000m,
                    )
                )

        result = SportsCourseRecommendResult(
            recommended_courses=recommended,
            avoided_courses=avoided,
            caution_notices=caution_notices,
            notices=notices,
        )
        self._audit(tags, result)
        return result

    def _audit(self, tags: Sequence[str], result: SportsCourseRecommendResult) -> None:
        """审计日志：仅记录推荐类型、输入标签摘要与输出条目计数（spec 4.4.3）。"""
        audit_logger.info(
            "选课建议完成 | 输入标签=%s | 推荐课目=%d | 避雷课目=%d",
            ",".join(tags) or "无",
            len(result.recommended_courses),
            len(result.avoided_courses),
        )


def create_default_sports_course_advisor() -> SportsCourseAdvisor:
    """装配默认体育课选课建议引擎：注入共享数据目录单例。"""
    return SportsCourseAdvisor(catalog=create_default_catalog())