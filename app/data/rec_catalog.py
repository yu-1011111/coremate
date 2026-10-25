"""推荐数据目录：解析四张数据表原文常量为只读数据资产（设计决策 D2）。

职责：load_records 解析 → A/R 前缀分流 → ID 唯一性校验（跨 A+R 全空间
与各表表内空间，重复条目拦截并保留首条，spec 5.1.1.4）→ 多值字段拆分
为 frozenset → 三档字段合法性归一（非法取值按未知档位处理并告警，
spec 5.1.3.2）→ 进程级缓存固化（同一进程仅解析一次，spec 5.1.1.3）。
"""

import logging
from typing import Optional

from app.data.loader import load_records
from app.data.raw_tables import (
    ACTIONS_TABLE_TEXT,
    DISHES_TABLE_TEXT,
    DRINKS_TABLE_TEXT,
    SPORT_COURSE_TABLE_TEXT,
)
from app.rules.rec_schemas import (
    ActionRecord,
    DishRecord,
    DrinkRecord,
    Level,
    RunCourseRecord,
    SportCourseRecord,
)

logger = logging.getLogger("coremate.data")


def _split_pipe_tags(value: Optional[str]) -> frozenset[str]:
    """按字面 `\\|`（反斜杠+竖线）拆分多值字段为标签集合。"""
    if not value:
        return frozenset()
    parts = (part.strip() for part in value.split("\\|"))
    return frozenset(part for part in parts if part)


def _split_comma_tags(value: Optional[str]) -> frozenset[str]:
    """按逗号拆分多值字段（菜品库 meal_time 的分隔符）。"""
    if not value:
        return frozenset()
    parts = (part.strip() for part in value.split(","))
    return frozenset(part for part in parts if part)


def _parse_level(value: Optional[str]) -> Optional[Level]:
    """三档字段归一：low/mid/high 以外取值按未知档位（None）处理并告警。"""
    if value is None:
        return None
    try:
        return Level(value.strip())
    except ValueError:
        logger.warning("三档字段出现非法取值，按未知档位处理: %r", value)
        return None


def _to_int(value: Optional[str]) -> Optional[int]:
    """数值字段解析：失败时按缺失处理并告警。"""
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        logger.warning("数值字段解析失败，按缺失处理: %r", value)
        return None


def _build_dish(record: dict[str, Optional[str]]) -> DishRecord:
    return DishRecord(
        dish_id=record["dish_id"] or "",
        name=record["name"] or "",
        meal_time=_split_comma_tags(record.get("meal_time")),
        purine_level=_parse_level(record.get("purine_level")),
        oil_level=_parse_level(record.get("oil_level")),
        salt_level=_parse_level(record.get("salt_level")),
        sugar_level=_parse_level(record.get("sugar_level")),
        avoid_for=_split_pipe_tags(record.get("avoid_for")),
        suitable_for=_split_pipe_tags(record.get("suitable_for")),
        audit_status=record.get("audit_status"),
        allergens=_split_pipe_tags(record.get("allergens")),
    )


def _build_drink(record: dict[str, Optional[str]]) -> DrinkRecord:
    return DrinkRecord(
        drink_id=record["drink_id"] or "",
        name=record["name"] or "",
        block_for=_split_pipe_tags(record.get("block_for")),
        caution_for=_split_pipe_tags(record.get("caution_for")),
        recommend_for=_split_pipe_tags(record.get("recommend_for")),
        swap_to=record.get("swap_to"),
        block_reason=record.get("block_reason"),
        caution_reason=record.get("caution_reason"),
        recommend_reason=record.get("recommend_reason"),
    )


def _build_action(record: dict[str, Optional[str]]) -> ActionRecord:
    return ActionRecord(
        action_id=record["action_id"] or "",
        name_cn=record["name_cn"] or "",
        type=record.get("type"),
        impact=_parse_level(record.get("impact")),
        est_minutes=_to_int(record.get("est_minutes")),
        scene_tags=_split_pipe_tags(record.get("scene_tags")),
        block_for=_split_pipe_tags(record.get("block_for")),
        caution_for=_split_pipe_tags(record.get("caution_for")),
        recommend_for=_split_pipe_tags(record.get("recommend_for")),
        fallback_cn=record.get("fallback_cn"),
        block_reason=record.get("block_reason"),
        caution_reason=record.get("caution_reason"),
        recommend_reason=record.get("recommend_reason"),
    )


def _build_run_course(record: dict[str, Optional[str]]) -> RunCourseRecord:
    return RunCourseRecord(
        course_id=record["action_id"] or "",
        name=record["name_cn"] or "",
        impact=_parse_level(record.get("impact")),
        duration_min=_to_int(record.get("duration_min")),
        scene_tags=_split_pipe_tags(record.get("scene_tags")),
        block_for=_split_pipe_tags(record.get("block_for")),
        caution_for=_split_pipe_tags(record.get("caution_for")),
        recommend_for=_split_pipe_tags(record.get("recommend_for")),
        fallback_cn=record.get("fallback_cn"),
        degrade_to=record.get("degrade_to"),
        weather_limit=_split_pipe_tags(record.get("weather_limit")),
        weather_swap=record.get("weather_swap"),
        block_reason=record.get("block_reason"),
        caution_reason=record.get("caution_reason"),
        recommend_reason=record.get("recommend_reason"),
    )


def _build_sport_course(record: dict[str, Optional[str]]) -> SportCourseRecord:
    return SportCourseRecord(
        course_id=record["course_id"] or "",
        name=record["name"] or "",
        category=record.get("category"),
        venue=record.get("venue"),
        block_for=_split_pipe_tags(record.get("block_for")),
        caution_for=_split_pipe_tags(record.get("caution_for")),
        recommend_for=_split_pipe_tags(record.get("recommend_for")),
        fitness_1000m=record.get("fitness_1000m"),
        block_reason=record.get("block_reason"),
        caution_reason=record.get("caution_reason"),
        recommend_reason=record.get("recommend_reason"),
    )


def _dedupe_by_id(items: list, id_getter, table_name: str) -> list:
    """表内（或 A+R 跨全空间）ID 唯一性校验：重复条目拦截并保留首条。"""
    seen: set[str] = set()
    deduped = []
    for item in items:
        key = id_getter(item)
        if key in seen:
            logger.warning("%s 存在重复 ID %s，保留首条并忽略后续", table_name, key)
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


class RecCatalog:
    """只读数据资产：五类记录的有序元组，按数据表原始条目顺序输出。"""

    def __init__(
        self,
        dishes: list[DishRecord],
        drinks: list[DrinkRecord],
        actions: list[ActionRecord],
        run_courses: list[RunCourseRecord],
        sport_courses: list[SportCourseRecord],
    ) -> None:
        self._dishes = tuple(dishes)
        self._drinks = tuple(drinks)
        self._actions = tuple(actions)
        self._run_courses = tuple(run_courses)
        self._sport_courses = tuple(sport_courses)
        if not self._dishes:
            logger.warning("菜品库条目数为 0，饮食推荐将返回数据暂不可用提示")
        if not self._drinks:
            logger.warning("饮品库条目数为 0，饮品判定将不可用")
        if not self._actions and not self._run_courses:
            logger.warning("动作库条目数为 0，运动推荐将返回数据暂不可用提示")
        if not self._sport_courses:
            logger.warning("体育课库条目数为 0，选课建议将返回数据暂不可用提示")

    @property
    def dishes(self) -> tuple[DishRecord, ...]:
        return self._dishes

    @property
    def drinks(self) -> tuple[DrinkRecord, ...]:
        return self._drinks

    @property
    def actions(self) -> tuple[ActionRecord, ...]:
        return self._actions

    @property
    def run_courses(self) -> tuple[RunCourseRecord, ...]:
        return self._run_courses

    @property
    def sport_courses(self) -> tuple[SportCourseRecord, ...]:
        return self._sport_courses


_DEFAULT_CATALOG: Optional[RecCatalog] = None


def _build_catalog() -> RecCatalog:
    raw_actions = load_records(ACTIONS_TABLE_TEXT)
    action_items = [_build_action(r) for r in raw_actions if (r.get("action_id") or "").startswith("A")]
    run_items = [_build_run_course(r) for r in raw_actions if (r.get("action_id") or "").startswith("R")]
    combined = _dedupe_by_id(
        action_items + run_items,
        lambda item: item.action_id if isinstance(item, ActionRecord) else item.course_id,
        "动作库(A+R 跨全空间)",
    )
    actions = [item for item in combined if isinstance(item, ActionRecord)]
    run_courses = [item for item in combined if isinstance(item, RunCourseRecord)]

    dishes = _dedupe_by_id([_build_dish(r) for r in load_records(DISHES_TABLE_TEXT)], lambda d: d.dish_id, "菜品库")
    drinks = _dedupe_by_id([_build_drink(r) for r in load_records(DRINKS_TABLE_TEXT)], lambda d: d.drink_id, "饮品库")
    sport_courses = _dedupe_by_id(
        [_build_sport_course(r) for r in load_records(SPORT_COURSE_TABLE_TEXT)], lambda c: c.course_id, "体育课库"
    )
    return RecCatalog(
        dishes=dishes,
        drinks=drinks,
        actions=actions,
        run_courses=run_courses,
        sport_courses=sport_courses,
    )


def create_default_catalog() -> RecCatalog:
    """进程级缓存工厂：同一进程仅解析一次且全程内容不变（spec 5.1.1.3）。"""
    global _DEFAULT_CATALOG
    if _DEFAULT_CATALOG is None:
        _DEFAULT_CATALOG = _build_catalog()
    return _DEFAULT_CATALOG