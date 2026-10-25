"""对话域数据契约：Agent 对话编排路由的领域对象与话术文案常量。

字段命名以需求规格（agent_chat_router spec.md）第 6 章为冻结契约；
内嵌结构引用级复用第二/三阶段既有模型（MatchedSignal / MoodResourceItem /
CrisisResponse / DietRecommendResult / ExerciseRecommendResult），
禁止另造平行结构（spec 4.5.3）。免责声明常量从第一阶段 schemas 导入，
单一事实源（spec 4.5.4）。
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from app.rules.mood_schemas import CrisisResponse, MatchedSignal, MoodResourceItem
from app.rules.rec_schemas import DietRecommendResult, ExerciseRecommendResult
from app.rules.schemas import DISCLAIMER_TEXT

MAX_TEXT_LENGTH = 500


class Route(str, Enum):
    """四级路由级别枚举（spec 6.2.1）。"""

    M4_CRISIS = "M4_CRISIS"
    MOOD_SUPPORT = "MOOD_SUPPORT"
    HEALTH_QUERY = "HEALTH_QUERY"
    SMALL_TALK = "SMALL_TALK"


class ChatRequest(BaseModel):
    """对话请求：user_id + text + 可选上轮上下文（quick_followup_fix spec 5.4）。

    三可选字段由前端携带（跨轮状态前端方案）：last_text 为上一条用户
    消息文本、shown_*_ids 为已展示推荐条目累计列表；既有调用方不传时
    行为与接入前逐字段一致（零 422 新增面）。
    """

    user_id: str = Field(min_length=1, description="用户标识（仅用于回显与审计）")
    text: str = Field(min_length=1, description="对话文本（≤500 字符，超长截断参与匹配）")
    last_text: Optional[str] = Field(default=None, description="上一条用户消息文本（追问换新解析来源）")
    shown_dish_ids: list[str] = Field(default_factory=list, description="已展示菜品条目累计（追问排除用）")
    shown_action_ids: list[str] = Field(default_factory=list, description="已展示动作条目累计（追问排除用）")


@dataclass(frozen=True)
class FollowupContext:
    """追问上下文（单轮生命周期冻结契约，禁止留存累积）。"""

    last_text: Optional[str] = None
    shown_dish_ids: tuple[str, ...] = ()
    shown_action_ids: tuple[str, ...] = ()


class ChatResponse(BaseModel):
    """对话聚合响应（十字段按 spec 6.2 顺序冻结，disclaimer 置尾）。

    diet/exercise 推荐字段为非 Optional 必填空结构（四类出口同一契约解析，
    spec 5.7.1.5a）；notices 只承载缺省参数与数据暂不可用提示，不复制
    内嵌引擎结果的 notices（决策 D5 字段语义边界）。
    """

    route: Route
    reply_text: str
    mood_level: str
    matched_signals: list[MatchedSignal] = Field(default_factory=list)
    mood_resources: list[MoodResourceItem] = Field(default_factory=list)
    diet_recommendations: DietRecommendResult = Field(default_factory=DietRecommendResult)
    exercise_recommendations: ExerciseRecommendResult = Field(default_factory=ExerciseRecommendResult)
    crisis: Optional[CrisisResponse] = None
    notices: list[str] = Field(default_factory=list)
    disclaimer: str = Field(default=DISCLAIMER_TEXT)


class RouterServiceError(Exception):
    """路由服务异常：危机链路与情绪判定失败的显式服务错误（决策 D3 fail-closed）。

    由路由器 raise、api/chat.py 捕获转 HTTP 500；禁止降级为任何其他路由出口。
    """


SMALL_TALK_SCRIPT = "你好呀，我是芯动管家。可以问我'午饭吃什么''怎么运动'，也可以随时和我聊聊最近的心情。"
MOOD_SUPPORT_SCRIPT = "听起来你最近过得不太轻松，先照顾好情绪，下面这些方法现在就能用："
BRIEF_ADVICE_DIET = "如果想知道吃什么，可以先试试这几样：{菜品列表}。"
BRIEF_ADVICE_EXERCISE = "想动一动的话，可以从这几个开始：{动作列表}。"
HEALTH_QUERY_SCRIPT = "根据你提到的情况（{标签摘要}），为你整理了{查询方向}的建议："
EXERCISE_DEFAULT_SCRIPT = "为你推荐 15 分钟的微运动方案："
FOLLOWUP_GUIDE_SCRIPT = "根据你提到的情况（{标签摘要}），再为你换几个刚才没展示过的{查询方向}建议："
FOLLOWUP_NO_CONTEXT_SCRIPT = "想看更多推荐，可以先说说你的情况，比如'午饭吃什么''怎么运动'。"
FOLLOWUP_EXHAUSTED_NOTICE = "符合条件的推荐刚才都展示过了，换个需求或场景（如换餐次、换时长）试试。"
FOLLOWUP_EXCLUDED_NOTICE = "已为你避开刚才展示过的 {数量} 条推荐。"
FALLBACK_REPLY = "我在的，可以和我聊聊健康或心情。"
DISH_CATALOG_MISS_REPLY = "系统暂未收录该菜品。为了您的健康，建议您选择清淡、少油盐的食物，或者拍照让我帮您识别。"
DISH_FORBIDDEN_TEMPLATE = "「{dish_name}」不太适合你当前的健康情况：{reason}。建议换成「{swap_to}」。"
DISH_SAFE_TEMPLATE = "「{dish_name}」没有命中你当前健康标签的禁忌，可以放心吃。"
DEFENSE_DRUG_REPLY = "抱歉，我不能提供处方药或医疗诊断，请前往校医院。"
DEFENSE_INJECTION_REPLY = "我是芯动管家，请勿进行系统攻击。"
DEFENSE_OFF_CAMPUS_REPLY = "建议合理规划课余时间，避免前往上述场所。"
DEFENSE_EXTREME_DIET_REPLY = "节食、暴食或催吐都会严重伤害身体。请先停止这些行为，规律三餐、均衡饮食；同时建议你联系学校心理中心，聊聊背后的压力，你不是一个人。"
BODY_TEST_ADVICE_TEMPLATE = "体测突击建议：建议以跑走交替与场地专项为主，考前避免高强度冲刺；天气不利时顺延或改室内替代。"
BODY_TEST_ADVICE_WEATHER_TEMPLATE = "体测突击建议：{天气条件句}，建议以跑走交替与场地专项为主，考前避免高强度冲刺；天气不利时顺延或改室内替代。"
BODY_TEST_WEATHER_OK = "当前天气适合户外训练"
BODY_TEST_WEATHER_BAD = "当前天气不太适合户外训练"
SERVICE_UNAVAILABLE_MESSAGE = "服务暂时不可用，请稍后再试"
CHAT_ROUTER_MISSING_MESSAGE = "对话服务暂不可用"
AVOID_FOOD_GUIDE_TEMPLATE = "根据你的健康情况（{标签摘要}），这些是你需要避开的红灯食物："
AVOID_FOOD_NO_TAGS_GUIDE = "先告诉我你的体检情况（比如尿酸高、血压高），我再帮你列出需要避开的红灯食物。"
AVOID_FOOD_EMPTY_NOTICE = "按你当前的健康情况，本轮没有命中需要避开的红灯禁忌。"
DRINK_BLOCKED_TEMPLATE = "「{饮品名}」不太适合你当前的健康情况：{理由}。建议改选「{替代}」。"
DRINK_CAUTION_TEMPLATE = "「{饮品名}」对你当前的情况需要留意：{理由}。"
DRINK_SAFE_TEMPLATE = "「{饮品名}」没有命中你当前健康标签的拦截，可以喝；注意含糖量，优先选无糖或低糖版本。"