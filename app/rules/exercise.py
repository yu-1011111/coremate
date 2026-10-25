"""运动推荐引擎：依据健康标签匹配动作库与跑步课表（spec 5.4）。

与饮食引擎同构范式（design 决策 D6）；超重/肥胖用户硬过滤 impact=high
动作并输出 fallback_cn 降级建议；跑步课表按 weather_limit 提供天气降级。
"""

import logging
from typing import Optional, Sequence

from app.data.rec_catalog import RecCatalog, create_default_catalog
from app.rules.rec_schemas import (
    ACTION_HARD_FILTERS,
    DATA_UNAVAILABLE_NOTICE,
    INVALID_WEATHER_NOTICE,
    NO_ACTION_FOR_SCENE_NOTICE,
    SHORT_TIME_NOTICE,
    SUPPRESSED_NOTICE,
    UNKNOWN_TAG_NOTICE,
    WEATHER_VALUES,
    ActionRecord,
    AvoidedAction,
    CautionNotice,
    ExerciseRecommendResult,
    FallbackSuggestion,
    HardFilterRule,
    RecommendedAction,
    RecommendedRunCourse,
    RunCourseRecord,
    Scene,
    extract_reason_for_tag,
)
from app.rules.tag_mapping import map_health_tags

audit_logger = logging.getLogger("coremate.audit")

RUN_COURSE_SCENES: tuple[Scene, ...] = (Scene.PLAYGROUND, Scene.TEST_CRASH)
HEALTHY_ACTION_REASON = "满足当前场景与时长约束，未命中任何禁忌条件"


class ExerciseRecommendEngine:
    """运动推荐引擎：场景圈定 → 硬过滤/block_for 避雷 → 时长+推荐 → 跑步课表判定。"""

    def __init__(self, catalog: RecCatalog) -> None:
        self._catalog = catalog

    def recommend(
        self,
        tags: Sequence[str],
        scene: Scene,
        available_minutes: int,
        weather: Optional[str] = None,
    ) -> ExerciseRecommendResult:
        """执行运动推荐：输入健康标签、场景、可用时长与可选天气。"""
        if not self._catalog.actions and not self._catalog.run_courses:
            result = ExerciseRecommendResult(notices=[DATA_UNAVAILABLE_NOTICE])
            self._audit(tags, scene, result)
            return result

        mapped = map_health_tags(tags)
        notices: list[str] = []
        if mapped.unknown_tags:
            notices.append(UNKNOWN_TAG_NOTICE)

        effective_weather: Optional[str] = None
        if weather is not None:
            if weather in WEATHER_VALUES:
                effective_weather = weather
            else:
                notices.append(INVALID_WEATHER_NOTICE)

        crowd = mapped.crowd_tags
        user_tag_set = frozenset(tags)
        candidates = [action for action in self._catalog.actions if scene.value in action.scene_tags]
        if not candidates:
            notices.append(NO_ACTION_FOR_SCENE_NOTICE)

        active_rules = [rule for rule in ACTION_HARD_FILTERS if rule.trigger_tags & user_tag_set] if crowd else []

        avoided: list[AvoidedAction] = []
        fallbacks: list[tuple[str, str]] = []
        if crowd:
            for action in candidates:
                reason, fallback = self._action_avoid_reason(action, active_rules, crowd, user_tag_set)
                if reason:
                    avoided.append(AvoidedAction(action_id=action.action_id, name=action.name_cn, reason=reason))
                    if fallback:
                        fallbacks.append(fallback)

        recommended: list[RecommendedAction] = []
        caution_notices: list[CautionNotice] = []
        avoided_ids = {item.action_id for item in avoided}
        if crowd:
            for action in candidates:
                if action.action_id in avoided_ids:
                    continue
                if action.est_minutes is None or action.est_minutes > available_minutes:
                    continue
                hit_recommend = action.recommend_for & crowd
                if hit_recommend:
                    tag = sorted(hit_recommend)[0]
                    reason = extract_reason_for_tag(action.recommend_reason, tag) or f"适合{tag}人群"
                    recommended.append(
                        RecommendedAction(
                            action_id=action.action_id,
                            name=action.name_cn,
                            reason=reason,
                            est_minutes=action.est_minutes,
                        )
                    )
                self._collect_action_caution(action, crowd, caution_notices)
        else:
            for action in candidates:
                if action.est_minutes is None or action.est_minutes > available_minutes:
                    continue
                recommended.append(
                    RecommendedAction(
                        action_id=action.action_id,
                        name=action.name_cn,
                        reason=HEALTHY_ACTION_REASON,
                        est_minutes=action.est_minutes,
                    )
                )

        if candidates and not recommended:
            feasible = [a for a in candidates if a.est_minutes is not None and a.est_minutes <= available_minutes]
            if not feasible:
                notices.append(SHORT_TIME_NOTICE)

        run_courses: list[RecommendedRunCourse] = []
        if scene in RUN_COURSE_SCENES:
            run_courses, course_avoided, course_cautions = self._judge_run_courses(
                crowd, active_rules, scene, available_minutes, effective_weather, user_tag_set
            )
            avoided.extend(course_avoided)
            caution_notices.extend(course_cautions)

        result = ExerciseRecommendResult(
            recommended_actions=recommended,
            avoided_actions=avoided,
            fallback_suggestions=[
                FallbackSuggestion(original=original, fallback=fallback) for original, fallback in fallbacks
            ],
            run_courses=run_courses,
            caution_notices=caution_notices,
            notices=notices,
        )
        self._audit(tags, scene, result)
        return result

    def recommend_suppressed(self) -> ExerciseRecommendResult:
        """M4 危机响应抑制构造：纯函数式空推荐结果（设计决策 D4 L2 层）。

        不触碰数据目录、不执行任何判定逻辑；notices 携带抑制说明，
        disclaimer 默认注入。recommend() 主方法行为零影响。
        """
        return ExerciseRecommendResult(
            recommended_actions=[],
            avoided_actions=[],
            fallback_suggestions=[],
            run_courses=[],
            caution_notices=[],
            notices=[SUPPRESSED_NOTICE],
        )

    @staticmethod
    def _action_avoid_reason(
        action: ActionRecord,
        active_rules: Sequence[HardFilterRule],
        crowd: frozenset[str],
        user_tag_set: frozenset[str] = frozenset(),
    ) -> tuple[Optional[str], Optional[tuple[str, str]]]:
        """动作避雷判定：硬过滤（impact=high）优先并绑定 fallback_cn 降级，其次 block_for。"""
        for rule in active_rules:
            if getattr(action, rule.target_field) is rule.level:
                hit_tags = sorted(rule.trigger_tags & user_tag_set)
                hit_tag = hit_tags[0] if hit_tags else sorted(rule.trigger_tags)[0]
                reason = rule.reason_template.format(tag=hit_tag)
                fallback = (action.name_cn, action.fallback_cn) if rule.bind_fallback and action.fallback_cn else None
                return reason, fallback
        hit_block = action.block_for & crowd
        if hit_block:
            tag = sorted(hit_block)[0]
            reason = extract_reason_for_tag(action.block_reason, tag) or f"命中 block_for 标注：{tag}"
            return reason, None
        return None, None

    @staticmethod
    def _collect_action_caution(
        action: ActionRecord, crowd: frozenset[str], cautions: list[CautionNotice]
    ) -> None:
        """动作谨慎提示：caution_for 命中且未避雷时输出提示，不阻止推荐（spec 5.4.1.10）。"""
        hit = action.caution_for & crowd
        if hit:
            tag = sorted(hit)[0]
            reason = extract_reason_for_tag(action.caution_reason, tag) or f"命中 caution_for 标注：{tag}"
            cautions.append(CautionNotice(kind="action", target=action.name_cn, reason=reason))

    def _judge_run_courses(
        self,
        crowd: frozenset[str],
        active_rules: Sequence[HardFilterRule],
        scene: Scene,
        available_minutes: int,
        weather: Optional[str],
        user_tag_set: frozenset[str],
    ) -> tuple[list[RecommendedRunCourse], list[AvoidedAction], list[CautionNotice]]:
        """跑步课表判定：避雷（block_for/硬过滤）→ 时长 → 推荐（含天气降级标记）。"""
        recommended: list[RecommendedRunCourse] = []
        avoided: list[AvoidedAction] = []
        cautions: list[CautionNotice] = []
        for course in self._catalog.run_courses:
            if scene.value not in course.scene_tags:
                continue
            if crowd:
                block_hit = course.block_for & crowd
                hard_hit = any(
                    getattr(course, rule.target_field) is rule.level
                    for rule in active_rules
                    if rule.target_field in RunCourseRecord.model_fields
                )
                if block_hit or hard_hit:
                    reason = self._course_avoid_reason(course, block_hit, hard_hit, active_rules, user_tag_set)
                    avoided.append(AvoidedAction(action_id=course.course_id, name=course.name, reason=reason))
                    continue
            if course.duration_min is None or course.duration_min > available_minutes:
                continue
            if crowd:
                hit_recommend = course.recommend_for & crowd
                if hit_recommend:
                    tag = sorted(hit_recommend)[0]
                    reason = extract_reason_for_tag(course.recommend_reason, tag) or f"适合{tag}人群"
                else:
                    reason = f"满足{scene.value}场景与 {available_minutes} 分钟时长约束"
                hit_caution = course.caution_for & crowd
                if hit_caution:
                    tag = sorted(hit_caution)[0]
                    caution_reason = extract_reason_for_tag(course.caution_reason, tag) or (
                        f"命中 caution_for 标注：{tag}"
                    )
                    cautions.append(CautionNotice(kind="run_course", target=course.name, reason=caution_reason))
            else:
                reason = f"满足{scene.value}场景与 {available_minutes} 分钟时长约束"
            weather_blocked = weather is not None and _weather_hit(course.weather_limit, weather)
            recommended.append(
                RecommendedRunCourse(
                    course_id=course.course_id,
                    name=course.name,
                    reason=reason,
                    duration_min=course.duration_min,
                    weather_blocked=weather_blocked,
                    weather_swap=course.weather_swap if weather_blocked else None,
                )
            )
        return recommended, avoided, cautions

    @staticmethod
    def _course_avoid_reason(
        course: RunCourseRecord,
        block_hit: frozenset[str],
        hard_hit: bool,
        active_rules: Sequence[HardFilterRule],
        user_tag_set: frozenset[str],
    ) -> str:
        """课表避雷理由：硬过滤优先，其次 block_for；可附 degrade_to 降级信息。"""
        if hard_hit:
            rule = next(rule for rule in active_rules if getattr(course, rule.target_field, None) is rule.level)
            hit_tags = sorted(rule.trigger_tags & user_tag_set)
            hit_tag = hit_tags[0] if hit_tags else sorted(rule.trigger_tags)[0]
            reason = rule.reason_template.format(tag=hit_tag)
        elif block_hit:
            tag = sorted(block_hit)[0]
            reason = extract_reason_for_tag(course.block_reason, tag) or f"命中 block_for 标注：{tag}"
        else:
            reason = "命中禁忌条件"
        if course.degrade_to:
            return f"{reason}；可降级为课表 {course.degrade_to}"
        return reason

    def _audit(self, tags: Sequence[str], scene: Scene, result: ExerciseRecommendResult) -> None:
        """审计日志：仅记录推荐类型、输入摘要与输出条目计数（spec 4.4.3）。"""
        audit_logger.info(
            "运动推荐完成 | 输入标签=%s | 场景=%s | 推荐动作=%d | 避雷动作=%d | 课表=%d",
            ",".join(tags) or "无",
            scene.value,
            len(result.recommended_actions),
            len(result.avoided_actions),
            len(result.run_courses),
        )


def _weather_hit(weather_limit: frozenset[str], weather: str) -> bool:
    """天气命中判定：请求"高温"命中"高温≥32℃"等带后缀字面值（前缀归一）。"""
    return any(limit == weather or limit.startswith(weather) for limit in weather_limit)


def create_default_exercise_engine() -> ExerciseRecommendEngine:
    """装配默认运动推荐引擎：注入共享数据目录单例。"""
    return ExerciseRecommendEngine(catalog=create_default_catalog())