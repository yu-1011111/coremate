"""心理域数据契约：以 Pydantic v2 定义心理干预模块的领域对象。

字段命名以需求规格（mood_intervention spec.md）第 6 章为冻结契约；
六类记录模型字段名与五张数据表附件表头逐字一致（spec 6.3-6.7）。
免责声明常量从第一阶段 schemas 导入，单一事实源（spec 4.5.4）。
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from app.rules.rec_schemas import SUPPRESSED_NOTICE, DietRecommendResult, ExerciseRecommendResult
from app.rules.schemas import DISCLAIMER_TEXT

NO_MOOD_ISSUE = "无情绪异常"

UNKNOWN_SC_NOTICE = "存在未识别的自查条目，已被忽略"
UNKNOWN_BEHAVIOR_NOTICE = "存在未识别的行为事件，已被忽略"
UNKNOWN_SELF_REPORT_NOTICE = "存在未识别的自报事件，已被忽略"
NO_VALID_SC_ERROR = "未识别任何有效自查条目"
EMPTY_SC_ERROR = "请至少勾选一条自查条目"
INVALID_DURATION_ERROR = "持续时长取值不合法"
EMPTY_SIGNAL_INPUT_ERROR = "请提供对话文本或行为事件"
LONG_DURATION_REFERRAL_NOTICE = "困扰持续较久，建议寻求专业支持"
KB_MISSING_NOTICE = "部分手册条目暂不可用"
REF_UNAVAILABLE_NOTICE = "引用对象暂不可用"
NO_RESOURCE_NOTICE = "当前档位暂无配套资源"
MOOD_DATA_UNAVAILABLE_NOTICE = "数据暂不可用"


WEIGHT_THRESHOLDS: tuple[tuple[int, int, str], ...] = (
    (1, 4, "M1"),
    (5, 8, "M2"),
    (9, 10**9, "M3"),
)

MOOD_RANK: dict[str, int] = {"M1": 1, "M2": 2, "M3": 3, "M4": 4}

SC_TO_SIGNAL_FALLBACK: dict[str, str] = {
    "SC1": "S09",
    "SC4": "S09",
    "SC3": "S10",
    "SC7": "S11",
    "SC9": "S12",
}


class MoodLevel(str, Enum):
    """情绪档位枚举（M1 低落 / M2 焦虑 / M3 烦躁 / M4 危机）。"""

    M1 = "M1"
    M2 = "M2"
    M3 = "M3"
    M4 = "M4"


class DurationLevel(str, Enum):
    """持续时长四档枚举（spec 6.1）。"""

    UNDER_2W = "<2 周"
    W2_TO_1M = "2 周-1 个月"
    M1_TO_3M = "1-3 个月"
    OVER_3M = "3 个月以上"


class BehaviorType(str, Enum):
    """行为事件类型五类枚举（spec 6.2.2，与信号表 keywords 对应）。"""

    LATE_NIGHT = "深夜活跃"
    LOW_STEPS = "低步数"
    EXERCISE_STALL = "运动停滞"
    DIET_SUPPRESS = "饮食压抑"
    NIGHT_WANDER = "夜间游荡"


class RefType(str, Enum):
    """CrossRef 跨库引用类型八类（spec 6.8）。"""

    ACTION = "action"
    RUN_COURSE = "run_course"
    AUDIO = "audio"
    FOOD = "food"
    DRINK = "drink"
    MOOD_KB = "mood_kb"
    CAMPUS = "campus"
    SCRIPT = "script"


class BehaviorEvent(BaseModel):
    """行为事件输入（统计职责归上层，引擎只做类型 + 天数阈值判定）。"""

    event_type: BehaviorType
    consecutive_days: int = Field(ge=1, description="连续天数")


class SelfcheckRequest(BaseModel):
    """便签自查请求：勾选编号必填非空，时长严格四档枚举。"""

    checked_sc_ids: list[str] = Field(min_length=1, description="勾选的自查条目编号")
    duration: DurationLevel = Field(description="持续时长档位")


class SignalRequest(BaseModel):
    """情绪信号识别请求：text 与 behavior_events 至少其一（引擎层跨字段校验）。"""

    text: str = Field(default="", description="对话文本")
    behavior_events: list[BehaviorEvent] = Field(default_factory=list, description="行为事件列表")
    self_reports: list[str] = Field(default_factory=list, description="自报事件 SC 编号列表")


class CrossRef(BaseModel):
    """跨库引用对象（设计决策 D2）：类型 + 对象标识 + 展示文本 + 可用性。"""

    model_config = ConfigDict(frozen=True)

    ref_type: RefType
    object_id: Optional[str] = None
    display_text: str
    available: bool = True


class MoodKbRecord(BaseModel):
    """心理手册记录：字段名与 mood_kb 表头逐字一致。"""

    model_config = ConfigDict(frozen=True)

    kb_id: str
    chapter: Optional[str] = None
    page: Optional[str] = None
    mood: frozenset[str]
    problem: Optional[str] = None
    method: Optional[str] = None
    howto: Optional[str] = None
    links: tuple[CrossRef, ...] = ()
    note: Optional[str] = None


class SelfcheckItemRecord(BaseModel):
    """便签自查记录：字段名与 mood_selfcheck 表头逐字一致。"""

    model_config = ConfigDict(frozen=True)

    sc_id: str
    text_original: Optional[str] = None
    mood_level: str
    kb_ids: frozenset[str]
    has_referral: bool = False
    referral_ids: frozenset[str]


class MoodSignalRecord(BaseModel):
    """触发信号记录：字段名与 mood_signals 表头逐字一致 + 两个派生字段（决策 D6）。"""

    model_config = ConfigDict(frozen=True)

    signal_id: str
    category: Optional[str] = None
    trigger_rule: Optional[str] = None
    keywords: frozenset[str]
    target_mood: Optional[str] = None
    weight: int = 0
    min_consecutive_days: int = 3
    sc_ids: frozenset[str] = frozenset()


class MoodResourceRecord(BaseModel):
    """资源记录：字段名与 mood_resources 表头逐字一致（ref 为单值引用）。"""

    model_config = ConfigDict(frozen=True)

    res_id: str
    mood_level: str
    res_type: Optional[str] = None
    ref: Optional[CrossRef] = None
    name: Optional[str] = None
    minutes: Optional[int] = None
    where: Optional[str] = None
    mechanism: Optional[str] = None
    priority: int = 0


class CampusResourceRecord(BaseModel):
    """校园求助资源记录：字段名与 campus_resources 表头逐字一致。"""

    model_config = ConfigDict(frozen=True)

    res_id: str
    name: str
    rtype: Optional[str] = None
    phone: Optional[str] = None
    hours: Optional[str] = None
    location: Optional[str] = None
    campus: Optional[str] = None
    note: Optional[str] = None
    verify_date: Optional[str] = None


class KbRecommendation(BaseModel):
    """手册推荐条目。"""

    kb_id: str
    method: Optional[str] = None
    howto: Optional[str] = None
    links: list[CrossRef] = Field(default_factory=list)


class ReferralInfo(BaseModel):
    """专业转介资源（名称 + 联系方式逐字取自 campus_resources）。"""

    res_id: str
    name: str
    contact: Optional[str] = None
    hours: Optional[str] = None
    note: Optional[str] = None


class MoodResourceItem(BaseModel):
    """档位情绪资源条目（含 ref 引用）。"""

    res_id: str
    mood_level: str
    res_type: Optional[str] = None
    name: Optional[str] = None
    minutes: Optional[int] = None
    where: Optional[str] = None
    mechanism: Optional[str] = None
    priority: int = 0
    ref: Optional[CrossRef] = None


class MatchedSignal(BaseModel):
    """命中信号（含命中具体关键词）。"""

    signal_id: str
    hit_keywords: list[str] = Field(default_factory=list)


class CrisisReferral(BaseModel):
    """危机转介项：统一结构承载 C 条目与引导语第六项（引导语 res_id=None）。"""

    res_id: Optional[str] = None
    name: str
    contact: Optional[str] = None
    hours: Optional[str] = None
    note: Optional[str] = None


class SelfcheckResult(BaseModel):
    """便签自查结果（spec 6.9；mood_level 限 M1-M3，M4 保护规则）。"""

    mood_level: str = Field(description="最终情绪档位（M1-M3，自查路径禁止输出 M4）")
    kb_recommendations: list[KbRecommendation] = Field(default_factory=list)
    referrals: list[ReferralInfo] = Field(default_factory=list)
    mood_resources: list[MoodResourceItem] = Field(default_factory=list)
    notices: list[str] = Field(default_factory=list)
    disclaimer: str = Field(default=DISCLAIMER_TEXT)


class SignalResult(BaseModel):
    """情绪信号识别结果（spec 6.10；crisis 仅 M4 时非空）。"""

    matched_signals: list[MatchedSignal] = Field(default_factory=list)
    total_weight: int = 0
    mood_level: str = NO_MOOD_ISSUE
    mood_resources: list[MoodResourceItem] = Field(default_factory=list)
    notices: list[str] = Field(default_factory=list)
    crisis: Optional["CrisisResponse"] = None
    disclaimer: str = Field(default=DISCLAIMER_TEXT)


class CrisisResponse(BaseModel):
    """M4 危机响应（设计决策 D7：字段声明顺序即序列化键顺序，免责声明置于收尾话术之后）。"""

    opening: str = Field(description="固定开场话术（逐字冻结）")
    referrals: list[CrisisReferral] = Field(default_factory=list)
    closing: str = Field(description="固定收尾话术（逐字冻结）")
    suppression: bool = Field(default=True, description="健康建议抑制标记")
    diet_recommendations: DietRecommendResult = Field(description="空饮食推荐结构")
    exercise_recommendations: ExerciseRecommendResult = Field(description="空运动推荐结构")
    disclaimer: str = Field(default=DISCLAIMER_TEXT)


class SelfcheckInvalidInputError(Exception):
    """自查输入整体非法（全未知编号），API 层转 400 语义错误。"""


class SignalEmptyInputError(Exception):
    """信号输入为空（text 与 behavior_events 均缺），API 层转 422。"""