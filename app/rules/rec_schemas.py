"""推荐域数据契约：以 Pydantic v2 定义饮食/运动/选课推荐的领域对象。

字段命名以需求规格（diet_exercise_rec spec.md）第 6 章为冻结契约；
五类记录模型字段名与四张数据表附件表头逐字一致（spec 5.1.1.1）。
免责声明常量从第一阶段 schemas 导入，单一事实源（spec 4.5.4）。
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from app.rules.schemas import DISCLAIMER_TEXT

UNKNOWN_TAG_NOTICE = "存在未识别的健康标签，已被忽略"
INVALID_WEATHER_NOTICE = "天气取值不合法，已忽略"
NO_DISH_FOR_MEAL_NOTICE = "当前餐次暂无可推荐菜品"
NO_ACTION_FOR_SCENE_NOTICE = "当前场景暂无可推荐动作"
FEW_RECOMMENDED_NOTICE = "当前餐次可选推荐较少"
NO_SAFE_DISH_NOTICE = "当前餐次在您的健康约束下暂无安全推荐，建议咨询食堂清淡窗口"
SHORT_TIME_NOTICE = "可用时长过短，暂无匹配动作，建议延长时长"
NO_COURSE_NOTICE = "在您当前健康约束下暂无合适体育课，建议咨询校医院与体育部"
DATA_UNAVAILABLE_NOTICE = "数据暂不可用"
DEFAULT_SWAP_TO = "白开水"
SUPPRESSED_NOTICE = "当前处于危机响应状态，健康建议已暂停"
HEALTHY_DISH_REASON = "未命中任何禁忌条件，可作常规选择"
REJECT_DISH_EXCLUDED = "审核未通过的菜品已排除在推荐之外"


class MealTime(str, Enum):
    """餐次枚举（spec 6.1.2）。"""

    BREAKFAST = "breakfast"
    LUNCH = "lunch"
    DINNER = "dinner"
    SNACK = "snack"
    SUPPER = "supper"


class Scene(str, Enum):
    """运动场景枚举（spec 6.1.3）。"""

    DORM_QUIET = "宿舍静音"
    TEST_CRASH = "体测突击"
    PLAYGROUND = "操场"
    BEDTIME_RELAX = "睡前放松"
    SEDENTARY_BREAK = "久坐起身"


class Level(str, Enum):
    """强度三档枚举（spec 领域术语）。"""

    LOW = "low"
    MID = "mid"
    HIGH = "high"


WEATHER_VALUES: tuple[str, ...] = ("晴", "雨", "雪", "霾", "大风", "高温", "低温")


class DietRecommendRequest(BaseModel):
    """饮食推荐请求：tags 宽松（体系外取值由引擎忽略并提示），meal 严格枚举。

    suppression 可选字段（默认 False）：M4 危机响应期间由上层置 True，
    路由层分流至引擎抑制构造返回空推荐（design 决策 D4 L3 层）。
    """

    tags: list[str] = Field(default_factory=list, description="健康标签列表")
    meal: MealTime = Field(description="餐次")
    suppression: bool = Field(default=False, description="危机响应健康建议抑制标记")


class ExerciseRecommendRequest(BaseModel):
    """运动推荐请求：scene 严格枚举、时长严格范围、weather 宽松透传。

    suppression 可选字段（默认 False）：M4 危机响应期间由上层置 True，
    路由层分流至引擎抑制构造返回空推荐（design 决策 D4 L3 层）。
    """

    tags: list[str] = Field(default_factory=list, description="健康标签列表")
    scene: Scene = Field(description="运动场景")
    available_minutes: int = Field(ge=1, le=240, description="可用时长（分钟）")
    weather: Optional[str] = Field(default=None, description="天气（宽松，非法取值由引擎忽略并提示）")
    suppression: bool = Field(default=False, description="危机响应健康建议抑制标记")


class SportsCourseRecommendRequest(BaseModel):
    """体育课选课建议请求。"""

    tags: list[str] = Field(default_factory=list, description="健康标签列表")


class DishRecord(BaseModel):
    """菜品记录：字段名与菜品库表头逐字一致。"""

    model_config = ConfigDict(frozen=True)

    dish_id: str
    name: str
    meal_time: frozenset[str]
    purine_level: Optional[Level] = None
    oil_level: Optional[Level] = None
    salt_level: Optional[Level] = None
    sugar_level: Optional[Level] = None
    avoid_for: frozenset[str]
    suitable_for: frozenset[str]
    audit_status: Optional[str] = None
    allergens: frozenset[str]


class DrinkRecord(BaseModel):
    """饮品记录：字段名与饮品库表头逐字一致。"""

    model_config = ConfigDict(frozen=True)

    drink_id: str
    name: str
    block_for: frozenset[str]
    caution_for: frozenset[str]
    recommend_for: frozenset[str]
    swap_to: Optional[str] = None
    block_reason: Optional[str] = None
    caution_reason: Optional[str] = None
    recommend_reason: Optional[str] = None


class ActionRecord(BaseModel):
    """动作记录：字段名与动作库表头逐字一致。"""

    model_config = ConfigDict(frozen=True)

    action_id: str
    name_cn: str
    type: Optional[str] = None
    impact: Optional[Level] = None
    est_minutes: Optional[int] = None
    scene_tags: frozenset[str]
    block_for: frozenset[str]
    caution_for: frozenset[str]
    recommend_for: frozenset[str]
    fallback_cn: Optional[str] = None
    block_reason: Optional[str] = None
    caution_reason: Optional[str] = None
    recommend_reason: Optional[str] = None


class RunCourseRecord(BaseModel):
    """跑步课表记录：源自动作库 R 前缀条目，字段名与表头逐字一致。"""

    model_config = ConfigDict(frozen=True)

    course_id: str
    name: str
    impact: Optional[Level] = None
    duration_min: Optional[int] = None
    scene_tags: frozenset[str]
    block_for: frozenset[str]
    caution_for: frozenset[str]
    recommend_for: frozenset[str]
    fallback_cn: Optional[str] = None
    degrade_to: Optional[str] = None
    weather_limit: frozenset[str]
    weather_swap: Optional[str] = None
    block_reason: Optional[str] = None
    caution_reason: Optional[str] = None
    recommend_reason: Optional[str] = None


class SportCourseRecord(BaseModel):
    """体育课记录：字段名与体育课库表头逐字一致。"""

    model_config = ConfigDict(frozen=True)

    course_id: str
    name: str
    category: Optional[str] = None
    venue: Optional[str] = None
    block_for: frozenset[str]
    caution_for: frozenset[str]
    recommend_for: frozenset[str]
    fitness_1000m: Optional[str] = None
    block_reason: Optional[str] = None
    caution_reason: Optional[str] = None
    recommend_reason: Optional[str] = None


class RecommendedDish(BaseModel):
    dish_id: str
    name: str
    reason: str


class AvoidedDish(BaseModel):
    dish_id: str
    name: str
    reason: str


class RecommendedDrink(BaseModel):
    drink_id: str
    name: str
    reason: str


class BlockedDrink(BaseModel):
    drink_id: str
    name: str
    reason: str
    swap_to: str


class RecommendedAction(BaseModel):
    action_id: str
    name: str
    reason: str
    est_minutes: Optional[int] = None
    cue: Optional[str] = None


class AvoidedAction(BaseModel):
    action_id: str
    name: str
    reason: str


class FallbackSuggestion(BaseModel):
    original: str
    fallback: str


class RecommendedRunCourse(BaseModel):
    course_id: str
    name: str
    reason: str
    duration_min: Optional[int] = None
    weather_blocked: bool = False
    weather_swap: Optional[str] = None


class RecommendedCourse(BaseModel):
    course_id: str
    name: str
    reason: str
    fitness_1000m: Optional[str] = None


class AvoidedCourse(BaseModel):
    course_id: str
    name: str
    reason: str


class CautionNotice(BaseModel):
    kind: str = Field(description="提示对象类别：drink/action/run_course/course")
    target: str = Field(description="提示对象名称")
    reason: str


class DietRecommendResult(BaseModel):
    """饮食推荐结果（spec 6.3）。"""

    recommended_dishes: list[RecommendedDish] = Field(default_factory=list)
    avoided_dishes: list[AvoidedDish] = Field(default_factory=list)
    recommended_drinks: list[RecommendedDrink] = Field(default_factory=list)
    blocked_drinks: list[BlockedDrink] = Field(default_factory=list)
    caution_notices: list[CautionNotice] = Field(default_factory=list)
    disclaimer: str = Field(default=DISCLAIMER_TEXT)
    notices: list[str] = Field(default_factory=list)


class ExerciseRecommendResult(BaseModel):
    """运动推荐结果（spec 6.4）。"""

    recommended_actions: list[RecommendedAction] = Field(default_factory=list)
    avoided_actions: list[AvoidedAction] = Field(default_factory=list)
    fallback_suggestions: list[FallbackSuggestion] = Field(default_factory=list)
    run_courses: list[RecommendedRunCourse] = Field(default_factory=list)
    caution_notices: list[CautionNotice] = Field(default_factory=list)
    disclaimer: str = Field(default=DISCLAIMER_TEXT)
    notices: list[str] = Field(default_factory=list)


class SportsCourseRecommendResult(BaseModel):
    """体育课选课建议结果（spec 6.5）。"""

    recommended_courses: list[RecommendedCourse] = Field(default_factory=list)
    avoided_courses: list[AvoidedCourse] = Field(default_factory=list)
    caution_notices: list[CautionNotice] = Field(default_factory=list)
    disclaimer: str = Field(default=DISCLAIMER_TEXT)
    notices: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class HardFilterRule:
    """硬过滤规则五元组：显式声明，禁止散落在匹配流程中隐式实现（design 决策 D5）。"""

    trigger_tags: frozenset[str]
    target_field: str
    level: Level
    reason_template: str
    bind_fallback: bool = False


DISH_HARD_FILTERS: tuple[HardFilterRule, ...] = (
    HardFilterRule(frozenset({"高尿酸"}), "purine_level", Level.HIGH, "高嘌呤食材，高尿酸人群应避免"),
    HardFilterRule(frozenset({"超重", "肥胖"}), "oil_level", Level.HIGH, "高油脂烹饪，{tag}人群应控制摄入"),
    HardFilterRule(frozenset({"高血压", "血压偏高"}), "salt_level", Level.HIGH, "高盐饮食，{tag}人群应避免"),
)

ACTION_HARD_FILTERS: tuple[HardFilterRule, ...] = (
    HardFilterRule(
        frozenset({"超重", "肥胖"}),
        "impact",
        Level.HIGH,
        "高冲击动作，{tag}人群膝踝关节承压风险高",
        bind_fallback=True,
    ),
)


def extract_reason_for_tag(reason_text: Optional[str], tag: str) -> Optional[str]:
    """按"标签：说明；标签：说明"结构提取指定标签的说明文本。

    数据表 reason 字段的通用格式：多段以中文分号分隔，每段以中文冒号
    区分标签与说明。未命中标签时返回 None，由调用方兜底转述。
    """
    if not reason_text:
        return None
    for segment in reason_text.split("；"):
        if "：" in segment:
            key, _, description = segment.partition("：")
            if key.strip() == tag:
                return description.strip()
    return None

class SpecificDishVerdict(BaseModel):
    """具体菜品判定结果（safety_scene_search 1.1）：命中身份 + 红/绿卡语义 + 平替。

    落位数据契约层（不复用 food_verdict.FoodVerdictResult——其缺菜品身份
    字段），避免 diet → food_verdict 层反转。
    """

    model_config = ConfigDict(frozen=True)

    matched: bool = False
    dish_id: Optional[str] = None
    dish_name: Optional[str] = None
    is_forbidden: bool = False
    reason: str = ""
    swap_to: Optional[str] = None
    match_kind: str = "not_found"
    notices: tuple[str, ...] = ()