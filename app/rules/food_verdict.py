"""菜品禁忌判定服务：识别菜名 → 菜品库匹配 → 禁忌判定 → 替代建议。

与 diet 引擎同构的禁忌判定（设计决策 D8）：引用复用 DISH_HARD_FILTERS
同一常量对象与 map_health_tags 同一映射函数，判定次序与
DietRecommendEngine._dish_avoid_reason 一致；零依赖 diet.py 代码，
语义一致性由 tests/test_real_api.py 的双实现锚定断言管控。
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

from app.data.rec_catalog import RecCatalog, create_default_catalog
from app.rules.rec_schemas import (
    DISH_HARD_FILTERS,
    UNKNOWN_TAG_NOTICE,
    DishRecord,

    MealTime,
)
from app.rules.tag_mapping import map_health_tags

audit_logger = logging.getLogger("coremate.audit")

UNKNOWN_FOOD_REASON = "未能识别图片中的菜品，请尝试换一张清晰的照片"
DISH_NOT_FOUND_REASON_TEMPLATE = "菜品库暂未收录「{food_name}」，暂无法按健康标签判定禁忌"
NO_TAGS_REASON = "未提供健康标签，按普适标准判定"
NOT_FORBIDDEN_REASON = "未命中您当前健康标签的禁忌标注"
SWAP_FALLBACK_DISH = "清炒西兰花"
DISH_MULTI_MATCH_NOTICE = "识别结果匹配多道菜品，已按第一项判定"
DISH_NOT_FOUND_NOTICE = "菜品库暂未收录该菜品"

UNKNOWN_FOOD_TEXT = "未知菜品"


@dataclass(frozen=True)
class FoodVerdictResult:
    """菜品判定结果（match_kind 供审计：exact/contains/contains_multi/not_found/unknown）。"""

    is_forbidden: bool = False
    reason: str = ""
    swap_to: Optional[str] = None
    match_kind: str = "not_found"
    notices: list[str] = field(default_factory=list)


class FoodVerdictService:
    """菜品禁忌判定：两级匹配 + 同构禁忌判定 + swap_to 选取。"""

    def __init__(self, catalog: RecCatalog) -> None:
        self._catalog = catalog

    def matches_catalog(self, text: str) -> bool:
        """库内存在性判定（只读、无副作用、无审计日志、零 notices）。

        匹配语义复用两级匹配规则：精确等值或双向子串包含（全库、
        不受餐次限制）；全库无命中返回 False。
        """
        for dish in self._catalog.dishes:
            if text == dish.name or dish.name in text or text in dish.name:
                return True
        return False

    def judge(self, food_name: str, tags: Sequence[str], meal: MealTime) -> FoodVerdictResult:
        """执行判定：菜名 → 匹配 → 禁忌 → 替代建议。"""
        if food_name == UNKNOWN_FOOD_TEXT:
            return FoodVerdictResult(
                is_forbidden=False,
                reason=UNKNOWN_FOOD_REASON,
                match_kind="unknown",
            )

        matched, match_kind = self._match_dish(food_name)
        if matched is None:
            result = FoodVerdictResult(
                is_forbidden=False,
                reason=DISH_NOT_FOUND_REASON_TEMPLATE.format(food_name=food_name),
                match_kind="not_found",
                notices=[DISH_NOT_FOUND_NOTICE],
            )
            self._audit(food_name, tags, result)
            return result

        notices: list[str] = []
        if match_kind == "contains_multi":
            notices.append(DISH_MULTI_MATCH_NOTICE)

        mapped = map_health_tags(tags)
        if mapped.unknown_tags:
            notices.append(UNKNOWN_TAG_NOTICE)
        user_tag_set = frozenset(tags)
        crowd = mapped.crowd_tags

        if not crowd:
            result = FoodVerdictResult(
                is_forbidden=False,
                reason=NO_TAGS_REASON,
                swap_to=None,
                match_kind=match_kind,
                notices=notices,
            )
            self._audit(food_name, tags, result)
            return result

        forbidden_reason = self._verdict_reason(matched, crowd, user_tag_set)
        if forbidden_reason is not None:
            swap_to = self._pick_swap(meal, crowd, user_tag_set)
            result = FoodVerdictResult(
                is_forbidden=True,
                reason=forbidden_reason,
                swap_to=swap_to,
                match_kind=match_kind,
                notices=notices,
            )
        else:
            result = FoodVerdictResult(
                is_forbidden=False,
                reason=NOT_FORBIDDEN_REASON,
                swap_to=None,
                match_kind=match_kind,
                notices=notices,
            )
        self._audit(food_name, tags, result)
        return result

    def _match_dish(self, food_name: str) -> tuple[Optional[DishRecord], str]:
        """两级菜品库匹配：精确 → 双向子串包含（全库不受餐次限制）。"""
        for dish in self._catalog.dishes:
            if food_name == dish.name:
                return dish, "exact"
        contains_hits = [
            dish
            for dish in self._catalog.dishes
            if food_name in dish.name or dish.name in food_name
        ]
        if not contains_hits:
            return None, "not_found"
        if len(contains_hits) == 1:
            return contains_hits[0], "contains"
        return contains_hits[0], "contains_multi"

    @staticmethod
    def _verdict_reason(
        dish: DishRecord, crowd: frozenset[str], user_tag_set: frozenset[str]
    ) -> Optional[str]:
        """禁忌判定（与 diet._dish_avoid_reason 同构）：硬过滤优先 → avoid_for。"""
        for rule in DISH_HARD_FILTERS:
            if rule.trigger_tags & user_tag_set and getattr(dish, rule.target_field) is rule.level:
                hit_tags = sorted(rule.trigger_tags & user_tag_set)
                hit_tag = hit_tags[0] if hit_tags else sorted(rule.trigger_tags)[0]
                return rule.reason_template.format(tag=hit_tag)
        hit = dish.avoid_for & crowd
        if hit:
            tag = sorted(hit)[0]
            return f"标注{tag}人群慎选"
        return None

    def _pick_swap(
        self, meal: MealTime, crowd: frozenset[str], user_tag_set: frozenset[str]
    ) -> str:
        """替代建议：同餐次 suitable_for 命中且自身不禁忌的第一条；兜底清炒西兰花。"""
        for dish in self._catalog.dishes:
            if meal.value not in dish.meal_time:
                continue
            if not (dish.suitable_for & crowd):
                continue
            if self._verdict_reason(dish, crowd, user_tag_set) is not None:
                continue
            return dish.name
        return SWAP_FALLBACK_DISH

    def _audit(self, food_name: str, tags: Sequence[str], result: FoodVerdictResult) -> None:
        """审计日志：菜名、匹配方式、判定结论与标签摘要（不含图片与凭证）。"""
        audit_logger.info(
            "菜品识别判定完成 | 菜名=%s | 匹配=%s | 判定=%s | 输入标签=%s | 替代=%s",
            food_name,
            result.match_kind,
            "禁忌" if result.is_forbidden else "通过",
            ",".join(tags) or "无",
            result.swap_to or "无",
        )


def create_default_food_verdict_service() -> FoodVerdictService:
    """装配默认判定服务：注入共享菜品库目录单例。"""
    return FoodVerdictService(catalog=create_default_catalog())