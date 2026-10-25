"""饮食推荐引擎：依据健康标签匹配菜品库与饮品库（spec 5.3）。

无状态线程安全、纯本地集合运算、无网络 IO；安全优先原则：同一菜品
同时命中推荐与避雷条件时必须归入避雷（spec 5.3.1.7）。
"""

import logging
from typing import Sequence

from app.data.rec_catalog import RecCatalog, create_default_catalog
from app.rules.rec_schemas import (
    DATA_UNAVAILABLE_NOTICE,
    DEFAULT_SWAP_TO,
    DISH_HARD_FILTERS,
    FEW_RECOMMENDED_NOTICE,
    HEALTHY_DISH_REASON,
    NO_DISH_FOR_MEAL_NOTICE,
    NO_SAFE_DISH_NOTICE,
    SUPPRESSED_NOTICE,
    UNKNOWN_TAG_NOTICE,
    AvoidedDish,
    BlockedDrink,
    CautionNotice,
    DietRecommendResult,
    DishRecord,
    HardFilterRule,
    MealTime,
    RecommendedDish,
    RecommendedDrink,
    SpecificDishVerdict,
    extract_reason_for_tag,
)
from app.rules.tag_mapping import map_health_tags

audit_logger = logging.getLogger("coremate.audit")

MIN_RECOMMENDED_DISHES = 3
SPECIFIC_DISH_SAFE_REASON = "未命中您当前健康标签的禁忌标注"
SPECIFIC_SWAP_FALLBACK = "清炒西兰花"


class DietRecommendEngine:
    """饮食推荐引擎：餐次圈定 → 硬过滤/avoid_for 避雷 → suitable_for 推荐 → 饮品三路判定。"""

    def __init__(self, catalog: RecCatalog) -> None:
        self._catalog = catalog

    def recommend(self, tags: Sequence[str], meal: MealTime) -> DietRecommendResult:
        """执行饮食推荐：输入健康标签与餐次，输出推荐/避雷/拦截清单与免责声明。"""
        if not self._catalog.dishes and not self._catalog.drinks:
            result = DietRecommendResult(notices=[DATA_UNAVAILABLE_NOTICE])
            self._audit(tags, result)
            return result

        mapped = map_health_tags(tags)
        notices: list[str] = []
        if mapped.unknown_tags:
            notices.append(UNKNOWN_TAG_NOTICE)

        crowd = mapped.crowd_tags
        user_tag_set = frozenset(tags)
        candidates = [dish for dish in self._catalog.dishes if meal.value in dish.meal_time]
        if not candidates:
            notices.append(NO_DISH_FOR_MEAL_NOTICE)

        active_rules = [rule for rule in DISH_HARD_FILTERS if rule.trigger_tags & user_tag_set] if crowd else []

        avoided: dict[str, AvoidedDish] = {}
        if crowd:
            for dish in candidates:
                reason = self._dish_avoid_reason(dish, active_rules, crowd, user_tag_set)
                if reason:
                    avoided[dish.dish_id] = AvoidedDish(dish_id=dish.dish_id, name=dish.name, reason=reason)

        recommended: list[RecommendedDish] = []
        if crowd:
            for dish in candidates:
                if dish.dish_id in avoided or dish.audit_status == "reject":
                    continue
                hit = dish.suitable_for & crowd
                if hit:
                    tag = sorted(hit)[0]
                    recommended.append(
                        RecommendedDish(dish_id=dish.dish_id, name=dish.name, reason=f"适合{tag}人群")
                    )
        else:
            for dish in candidates:
                if dish.audit_status == "reject":
                    continue
                recommended.append(
                    RecommendedDish(dish_id=dish.dish_id, name=dish.name, reason=HEALTHY_DISH_REASON)
                )

        if crowd and candidates and not recommended:
            notices.append(NO_SAFE_DISH_NOTICE)
        elif candidates and 0 < len(recommended) < MIN_RECOMMENDED_DISHES:
            notices.append(FEW_RECOMMENDED_NOTICE)

        blocked_drinks, recommended_drinks, caution_notices = self._judge_drinks(crowd)

        result = DietRecommendResult(
            recommended_dishes=recommended,
            avoided_dishes=list(avoided.values()),
            recommended_drinks=recommended_drinks,
            blocked_drinks=blocked_drinks,
            caution_notices=caution_notices,
            notices=notices,
        )
        self._audit(tags, result)
        return result

    @staticmethod
    def _dish_avoid_reason(
        dish: DishRecord,
        active_rules: Sequence[HardFilterRule],
        crowd: frozenset[str],
        user_tag_set: frozenset[str] = frozenset(),
    ) -> str | None:
        """菜品避雷理由：硬过滤模板优先，其次 avoid_for 事实性转述。"""
        for rule in active_rules:
            if getattr(dish, rule.target_field) is rule.level:
                hit_tags = sorted(rule.trigger_tags & user_tag_set)
                hit_tag = hit_tags[0] if hit_tags else sorted(rule.trigger_tags)[0]
                return rule.reason_template.format(tag=hit_tag)
        hit = dish.avoid_for & crowd
        if hit:
            tag = sorted(hit)[0]
            return f"标注{tag}人群慎选"
        return None

    def _judge_drinks(
        self, crowd: frozenset[str]
    ) -> tuple[list[BlockedDrink], list[RecommendedDrink], list[CautionNotice]]:
        """饮品三路判定：block 拦截（附替代）> caution 提示（不推荐）> recommend 推荐。"""
        blocked: list[BlockedDrink] = []
        recommended: list[RecommendedDrink] = []
        cautions: list[CautionNotice] = []
        if not crowd:
            return blocked, recommended, cautions
        for drink in self._catalog.drinks:
            hit_block = drink.block_for & crowd
            if hit_block:
                tag = sorted(hit_block)[0]
                reason = extract_reason_for_tag(drink.block_reason, tag) or f"命中 block_for 标注：{tag}"
                swap_to = drink.swap_to or DEFAULT_SWAP_TO
                blocked.append(
                    BlockedDrink(
                        drink_id=drink.drink_id,
                        name=drink.name,
                        reason=f"{reason}，建议改选：{swap_to}",
                        swap_to=swap_to,
                    )
                )
                continue
            hit_caution = drink.caution_for & crowd
            if hit_caution:
                tag = sorted(hit_caution)[0]
                reason = extract_reason_for_tag(drink.caution_reason, tag) or f"命中 caution_for 标注：{tag}"
                cautions.append(CautionNotice(kind="drink", target=drink.name, reason=reason))
                continue
            hit_recommend = drink.recommend_for & crowd
            if hit_recommend:
                tag = sorted(hit_recommend)[0]
                reason = extract_reason_for_tag(drink.recommend_reason, tag) or f"适合{tag}人群"
                recommended.append(
                    RecommendedDrink(drink_id=drink.drink_id, name=drink.name, reason=reason)
                )
        return blocked, recommended, cautions

    def recommend_suppressed(self) -> DietRecommendResult:
        """M4 危机响应抑制构造：纯函数式空推荐结果（设计决策 D4 L2 层）。

        不触碰数据目录、不执行任何判定逻辑；notices 携带抑制说明，
        disclaimer 默认注入。recommend() 主方法行为零影响。
        """
        return DietRecommendResult(
            recommended_dishes=[],
            avoided_dishes=[],
            recommended_drinks=[],
            blocked_drinks=[],
            caution_notices=[],
            notices=[SUPPRESSED_NOTICE],
        )

    def check_specific_dish(self, dish_name: str, user_tags: Sequence[str]) -> SpecificDishVerdict:
        """具体菜品判定（safety_scene_search 1.5）：定位 → 禁忌 → 中性餐次平替。

        独立新增方法：不被 recommend() / recommend_suppressed() 调用，
        两者调用图与逐行逻辑零变化；纯本地只读，零网络。
        匹配语义复用两级匹配（精确优先 → 双向子串包含，全库免餐次）。
        """
        matched = self._match_dish_by_name(dish_name)
        if matched is None:
            return SpecificDishVerdict(matched=False, match_kind="not_found")
        dish, match_kind = matched

        crowd = map_health_tags(user_tags).crowd_tags
        user_tag_set = frozenset(user_tags)
        if not crowd:
            return SpecificDishVerdict(
                matched=True,
                dish_id=dish.dish_id,
                dish_name=dish.name,
                is_forbidden=False,
                reason=SPECIFIC_DISH_SAFE_REASON,
                match_kind=match_kind,
            )

        active_rules = [rule for rule in DISH_HARD_FILTERS if rule.trigger_tags & user_tag_set]
        reason = self._dish_avoid_reason(dish, active_rules, crowd, user_tag_set)
        if reason:
            return SpecificDishVerdict(
                matched=True,
                dish_id=dish.dish_id,
                dish_name=dish.name,
                is_forbidden=True,
                reason=reason,
                swap_to=self._pick_specific_swap(crowd, user_tag_set),
                match_kind=match_kind,
            )
        return SpecificDishVerdict(
            matched=True,
            dish_id=dish.dish_id,
            dish_name=dish.name,
            is_forbidden=False,
            reason=SPECIFIC_DISH_SAFE_REASON,
            match_kind=match_kind,
        )

    def _match_dish_by_name(self, dish_name: str) -> tuple[DishRecord, str] | None:
        """两级匹配定位库内菜品（语义同构 food_verdict 但保持引擎自包含）。"""
        for dish in self._catalog.dishes:
            if dish_name == dish.name:
                return dish, "exact"
        contains_hits = [
            dish
            for dish in self._catalog.dishes
            if dish_name in dish.name or dish.name in dish_name
        ]
        if not contains_hits:
            return None
        if len(contains_hits) == 1:
            return contains_hits[0], "contains"
        return contains_hits[0], "contains_multi"

    def _pick_specific_swap(self, crowd: frozenset[str], user_tag_set: frozenset[str]) -> str:
        """中性缺省餐次平替：午餐 suitable_for 命中且自身不禁忌首条；兜底常量。"""
        for dish in self._catalog.dishes:
            if MealTime.LUNCH.value not in dish.meal_time:
                continue
            if not (dish.suitable_for & crowd):
                continue
            active_rules = [rule for rule in DISH_HARD_FILTERS if rule.trigger_tags & user_tag_set]
            if self._dish_avoid_reason(dish, active_rules, crowd, user_tag_set) is not None:
                continue
            return dish.name
        return SPECIFIC_SWAP_FALLBACK

    def _audit(self, tags: Sequence[str], result: DietRecommendResult) -> None:
        """审计日志：仅记录推荐类型、输入标签摘要与输出条目计数（spec 4.4.3）。"""
        audit_logger.info(
            "饮食推荐完成 | 输入标签=%s | 推荐菜品=%d | 避雷菜品=%d | 推荐饮品=%d | 拦截饮品=%d",
            ",".join(tags) or "无",
            len(result.recommended_dishes),
            len(result.avoided_dishes),
            len(result.recommended_drinks),
            len(result.blocked_drinks),
        )


def create_default_diet_engine() -> DietRecommendEngine:
    """装配默认饮食推荐引擎：注入共享数据目录单例。"""

    return DietRecommendEngine(catalog=create_default_catalog())