"""意图与实体解析器：纯本地关键词/规则匹配（spec 5.2，不调用大模型）。

关键词映射表以显式常量独立维护、与路由判定逻辑分离（spec 4.4.1），
与既有 tag_mapping.py 同构组织；解析器为纯函数无副作用、相同文本
返回值完全相等（决策 D12 确定性基础）。
"""

import logging
import re
from typing import Optional

from pydantic import BaseModel, Field

from app.rules.rec_schemas import MealTime, Scene, WEATHER_VALUES
from app.rules.tag_mapping import KNOWN_HEALTH_TAGS

logger = logging.getLogger("coremate.data")

TAG_KEYWORDS: dict[str, tuple[str, ...]] = {
    "高尿酸": ("尿酸高", "尿酸有点高", "尿酸偏高", "尿酸超标", "痛风"),
    "高胆固醇": ("胆固醇高", "胆固醇有点高", "胆固醇偏高", "胆固醇超标"),
    "高甘油三酯": ("甘油三酯高", "甘油三酯有点高", "甘油三酯偏高", "甘油三酯超标"),
    "低密度脂蛋白偏高": ("低密度脂蛋白高", "低密度脂蛋白偏高", "坏胆固醇高"),
    "高密度脂蛋白偏低": ("高密度脂蛋白低", "高密度脂蛋白偏低", "好胆固醇低"),
    "高血糖风险": ("血糖高", "血糖有点高", "血糖偏高", "血糖超标", "血糖不好"),
    "空腹血糖受损": ("空腹血糖高", "空腹血糖偏高", "空腹血糖受损"),
    "高血压": ("高血压",),
    "血压偏高": ("血压高", "血压有点高", "血压偏高"),
    "超重": ("超重", "有点胖", "偏胖", "很胖", "太胖", "好胖", "体重超标"),
    "肥胖": ("肥胖",),
    "偏瘦": ("偏瘦", "太瘦", "很瘦", "瘦弱", "体重过轻"),
    "贫血风险": ("贫血",),
    "幽门螺杆菌阳性": ("幽门螺杆菌", "幽门螺旋杆菌", "螺杆菌阳性"),
    "肠胃敏感": ("胃病", "反酸", "胃不好", "肠胃不好", "胃不舒服"),
}

MEAL_KEYWORDS: dict[str, MealTime] = {
    "早餐": MealTime.BREAKFAST,
    "早饭": MealTime.BREAKFAST,
    "早点": MealTime.BREAKFAST,
    "早上吃": MealTime.BREAKFAST,
    "午餐": MealTime.LUNCH,
    "午饭": MealTime.LUNCH,
    "中午吃": MealTime.LUNCH,
    "中午饭": MealTime.LUNCH,
    "晚餐": MealTime.DINNER,
    "晚饭": MealTime.DINNER,
    "晚上吃": MealTime.DINNER,
    "零食": MealTime.SNACK,
    "加餐": MealTime.SNACK,
    "下午茶": MealTime.SNACK,
    "夜宵": MealTime.SUPPER,
    "宵夜": MealTime.SUPPER,
    "深夜吃": MealTime.SUPPER,
}

SCENE_KEYWORDS: dict[str, Scene] = {
    "宿舍": Scene.DORM_QUIET,
    "寝室": Scene.DORM_QUIET,
    "室内": Scene.DORM_QUIET,
    "在家": Scene.DORM_QUIET,
    "体测": Scene.TEST_CRASH,
    "体能测试": Scene.TEST_CRASH,
    "操场": Scene.PLAYGROUND,
    "跑道": Scene.PLAYGROUND,
    "跑步": Scene.PLAYGROUND,
    "睡前": Scene.BEDTIME_RELAX,
    "睡觉前": Scene.BEDTIME_RELAX,
    "久坐": Scene.SEDENTARY_BREAK,
    "课间": Scene.SEDENTARY_BREAK,
    "坐久了": Scene.SEDENTARY_BREAK,
}

WEATHER_KEYWORDS: dict[str, str] = {
    "下雨": "雨",
    "雨天": "雨",
    "下雪": "雪",
    "雪天": "雪",
    "雾霾": "霾",
    "刮大风": "大风",
    "高温": "高温",
    "太热": "高温",
    "降温": "低温",
    "太冷": "低温",
    "晴天": "晴",
}

DIET_INTENT_KEYWORDS: tuple[str, ...] = (
    "吃什么", "吃啥", "怎么吃", "能吃", "能吃什么", "吃点", "喝点", "推荐菜", "可以吃",
)

EXERCISE_INTENT_KEYWORDS: tuple[str, ...] = (
    "运动", "锻炼", "健身", "怎么动", "练什么", "怎么练", "跑步", "慢跑", "马拉松",
    "体能", "体测", "800米", "1000米", "练腹肌", "瘦腿", "减肥",
)

OUTDOOR_RUNNING_KEYWORDS: tuple[str, ...] = ("操场跑步", "跑步", "慢跑", "马拉松")
INDOOR_WORKOUT_KEYWORDS: tuple[str, ...] = ("宿舍", "寝室", "室内", "在家")

FOLLOWUP_KEYWORDS: tuple[str, ...] = ("还有其他", "再推荐", "换一个", "再来几个", "别的动作", "别的菜")
FOLLOWUP_TARGET_EXERCISE: tuple[str, ...] = ("动作", "运动", "练")
FOLLOWUP_TARGET_DIET: tuple[str, ...] = ("菜", "饭", "吃")

DISH_QUERY_KEYWORDS: tuple[str, ...] = (
    "怎么做", "做法", "怎么做好吃", "好吃吗", "好不好吃", "能不能吃", "可不可以吃",
    "可以吃吗", "能吃吗", "这个菜", "这道菜", "这菜", "适不适合吃",
)

SPECIFIC_FOOD_QUERY_KEYWORDS: tuple[str, ...] = (
    "适不适合吃", "可不可以吃", "能不能吃", "可以吃吗", "我想吃", "能吃吗", "想吃", "可以吃", "能吃",
)
GENERIC_FOOD_WORDS: tuple[str, ...] = ("什么", "啥", "点东西", "哪个", "哪种", "哪些")

FITNESS_TEST_KEYWORDS: tuple[str, ...] = (
    "体测", "体能测试", "800 米", "800米", "1000 米", "1000米", "长跑",
    "八百", "一千", "考八百", "拉练", "跑圈",
)
THEORY_QUESTION_WORDS: tuple[str, ...] = ("怎么", "如何", "怎样", "技巧", "注意")
THEORY_TOPIC_WORDS: tuple[str, ...] = ("呼吸", "热身", "姿势", "跑姿", "配速", "岔气", "拉伸", "减量")
NEGATION_KEYWORDS: tuple[str, ...] = ("没病", "不过敏", "健康", "不高", "没有", "不是", "不痛", "无", "没")
NEGATION_LOOKBACK_CHARS = 4

AVOID_FOOD_QUERY_KEYWORDS: tuple[str, ...] = (
    "不能吃什么", "禁止吃啥", "不能吃啥", "避开什么", "禁忌是什么", "有什么忌口",
)
DRINK_COLLOQUIAL_KEYWORDS: tuple[str, ...] = ("肥宅快乐水", "波波", "奶茶", "果茶")
CONSTIPATION_QUERY_KEYWORDS: tuple[str, ...] = ("拉不出屎", "拉不出来", "便秘")
CORE_TRAINING_KEYWORDS: tuple[str, ...] = ("练腹肌",)
SLIMMING_KEYWORDS: tuple[str, ...] = ("瘦腿", "减肥")
HYPOTHETICAL_MARKERS: tuple[str, ...] = ("如果", "假如", "要是", "万一")
HYPOTHETICAL_CONSEQUENCE_WORDS: tuple[str, ...] = (
    "低血糖", "加重", "痛风发作", "升尿酸", "伤胃", "长胖", "反弹", "心梗", "晕倒",
)
HYPOTHETICAL_QUESTION_MARKS: tuple[str, ...] = (
    "吗", "么", "？", "会怎样", "会怎么样", "怎么办", "会不会", "会加重",
)

DURATION_PATTERN = re.compile(r"(\d+)\s*分钟")

DEFAULT_MINUTES = 15
MIN_MINUTES = 1
MAX_MINUTES = 240


SCENE_DEFAULT_NOTICE = "未识别运动场景，已默认按操场推荐"


MEAL_DISPLAY: dict[MealTime, str] = {
    MealTime.BREAKFAST: "早餐",
    MealTime.LUNCH: "午餐",
    MealTime.DINNER: "晚餐",
    MealTime.SNACK: "加餐",
    MealTime.SUPPER: "夜宵",
}

if not set(TAG_KEYWORDS) <= KNOWN_HEALTH_TAGS:
    logger.warning("TAG_KEYWORDS 存在 14 标签体系外的键，运行期按未提取处理")
if not all(value in MealTime for value in MEAL_KEYWORDS.values()):
    logger.warning("MEAL_KEYWORDS 存在非法餐次枚举值")
if not all(value in Scene for value in SCENE_KEYWORDS.values()):
    logger.warning("SCENE_KEYWORDS 存在非法场景枚举值")
if not all(value in WEATHER_VALUES for value in WEATHER_KEYWORDS.values()):
    logger.warning("WEATHER_KEYWORDS 存在非法天气取值")


class IntentParseResult(BaseModel):
    """意图解析结果（spec 6.3 内部冻结契约：七类要素 + notices）。"""

    tags: list[str] = Field(default_factory=list)
    meal: Optional[MealTime] = None
    scene: Optional[Scene] = None
    available_minutes: Optional[int] = None
    minutes_defaulted: bool = False
    weather: Optional[str] = None
    diet_intent: bool = False
    exercise_intent: bool = False
    followup_intent: bool = False
    dish_query_intent: bool = False
    specific_food_intent: bool = False
    food_candidate: Optional[str] = None
    outdoor_running_intent: bool = False
    indoor_workout_intent: bool = False
    fitness_test_intent: bool = False
    theory_query_intent: bool = False
    avoid_food_intent: bool = False
    drink_query_intent: bool = False
    hypothetical_query_intent: bool = False
    constipation_query_intent: bool = False
    core_training_intent: bool = False
    slimming_intent: bool = False
    notices: list[str] = Field(default_factory=list)


class IntentParser:
    """意图与实体解析器：意图判定 → 短路 → 标签/餐次/场景/时长/天气提取。

    决策 D8 单入口 parse：无任何健康意图时立即返回（实体提取代码不执行，
    spec 5.2.1.10）；纯函数无副作用不抛异常。
    """

    def parse(self, text: str) -> IntentParseResult:
        """解析对话文本的意图要素（含追问意图标志）。"""
        meal_hits = [(text.find(keyword), keyword) for keyword in MEAL_KEYWORDS if keyword in text]
        meal_hit = min(meal_hits) if meal_hits else None
        diet_intent = meal_hit is not None or any(keyword in text for keyword in DIET_INTENT_KEYWORDS)
        exercise_intent = any(keyword in text for keyword in EXERCISE_INTENT_KEYWORDS)
        followup_intent = any(keyword in text for keyword in FOLLOWUP_KEYWORDS)
        dish_query_intent = any(keyword in text for keyword in DISH_QUERY_KEYWORDS)
        avoid_food_intent = any(keyword in text for keyword in AVOID_FOOD_QUERY_KEYWORDS)
        food_candidate = self._extract_food_candidate(text)
        if avoid_food_intent:
            food_candidate = None
            dish_query_intent = False
        specific_food_intent = food_candidate is not None
        outdoor_running_intent = any(keyword in text for keyword in OUTDOOR_RUNNING_KEYWORDS)
        indoor_workout_intent = any(keyword in text for keyword in INDOOR_WORKOUT_KEYWORDS)
        fitness_test_intent = any(keyword in text for keyword in FITNESS_TEST_KEYWORDS)
        theory_query_intent = any(
            keyword in text for keyword in THEORY_QUESTION_WORDS
        ) and any(keyword in text for keyword in THEORY_TOPIC_WORDS)
        drink_query_intent = any(keyword in text for keyword in DRINK_COLLOQUIAL_KEYWORDS)
        constipation_query_intent = any(keyword in text for keyword in CONSTIPATION_QUERY_KEYWORDS)
        core_training_intent = any(keyword in text for keyword in CORE_TRAINING_KEYWORDS)
        slimming_intent = any(keyword in text for keyword in SLIMMING_KEYWORDS)
        hypothetical_query_intent = (
            any(keyword in text for keyword in HYPOTHETICAL_MARKERS)
            or any(keyword in text for keyword in HYPOTHETICAL_CONSEQUENCE_WORDS)
        ) and any(keyword in text for keyword in HYPOTHETICAL_QUESTION_MARKS)

        exercise_active = exercise_intent or fitness_test_intent

        if (
            not diet_intent
            and not exercise_intent
            and not followup_intent
            and not dish_query_intent
            and not specific_food_intent
            and not fitness_test_intent
            and not theory_query_intent
            and not avoid_food_intent
            and not drink_query_intent
            and not hypothetical_query_intent
            and not constipation_query_intent
            and not core_training_intent
            and not slimming_intent
        ):
            return IntentParseResult()

        notices: list[str] = []
        tags = self._extract_tags(text)
        meal = self._extract_meal(diet_intent, meal_hit, notices)
        scene = self._extract_scene(exercise_active, text, notices)
        minutes, minutes_defaulted = self._extract_minutes(exercise_active, text)
        weather = self._extract_weather(exercise_active, text)

        return IntentParseResult(
            tags=tags,
            meal=meal,
            scene=scene,
            available_minutes=minutes,
            minutes_defaulted=minutes_defaulted,
            weather=weather,
            diet_intent=diet_intent,
            exercise_intent=exercise_intent,
            followup_intent=followup_intent,
            dish_query_intent=dish_query_intent,
            specific_food_intent=specific_food_intent,
            food_candidate=food_candidate,
            outdoor_running_intent=outdoor_running_intent,
            indoor_workout_intent=indoor_workout_intent,
            fitness_test_intent=fitness_test_intent,
            theory_query_intent=theory_query_intent,
            avoid_food_intent=avoid_food_intent,
            drink_query_intent=drink_query_intent,
            hypothetical_query_intent=hypothetical_query_intent,
            constipation_query_intent=constipation_query_intent,
            core_training_intent=core_training_intent,
            slimming_intent=slimming_intent,
            notices=notices,
        )

    @staticmethod
    def _extract_food_candidate(text: str) -> Optional[str]:
        """具体菜名候选提取纯函数（1.4）：最长触发词剥离 → 取最长剩余片段。

        触发词可位于菜名前（"我想吃红烧肉"）或后（"红烧肉能吃吗"），
        故剥离后取前后两段中较长的候选；长度 < 2 或含泛指词 → 提取失败。
        同输入同输出、无副作用。
        """
        hits = [keyword for keyword in SPECIFIC_FOOD_QUERY_KEYWORDS if keyword in text]
        if not hits:
            return None
        keyword = max(hits, key=len)
        idx = text.find(keyword)
        if idx < 0:
            return None

        def _clean(segment: str) -> str:
            return segment.strip().strip("，。！？!?、,.：: ")

        after_segment = _clean(text[idx + len(keyword):])
        before_segment = _clean(text[:idx])
        # 触发词后的片段优先（"我想吃红烧肉"），其后无菜名时取触发词前（"红烧肉能吃吗"）
        for candidate in (after_segment, before_segment):
            if len(candidate) < 2:
                continue
            if any(word in candidate for word in GENERIC_FOOD_WORDS):
                continue
            return candidate
        return None

    @staticmethod
    def _extract_tags(text: str) -> list[str]:
        """健康标签提取：长词优先 + 区间包含抑制 + 前置窗口否定抑制。

        全部（标签, 关键词）对按关键词长度降序扫描；更短关键词命中时若
        其区间完全落入已命中区间（长词子串）则跳过——由长词决定标签；
        多标签并存全保留。否定抑制（llm_hardening_fitness D-否定）：命中
        位置前置 ≤4 字符窗口内出现任一否定词时跳过该命中（仅窗口内判定，
        不作用于命中区间之后）；异常 fail-open 按既有标签处理。
        """
        pairs = sorted(
            ((tag, keyword) for tag, keywords in TAG_KEYWORDS.items() for keyword in keywords),
            key=lambda pair: len(pair[1]),
            reverse=True,
        )
        hit_tags: list[str] = []
        hit_spans: list[tuple[int, int]] = []
        for tag, keyword in pairs:
            start = text.find(keyword)
            if start < 0:
                continue
            end = start + len(keyword)
            if any(s <= start and end <= e for s, e in hit_spans):
                continue
            try:
                window = text[max(0, start - NEGATION_LOOKBACK_CHARS):start]
                if any(negation in window for negation in NEGATION_KEYWORDS):
                    continue
            except Exception as exc:
                logger.warning(
                    "否定词抑制异常（类型=%s），按既有标签处理", type(exc).__name__
                )
            hit_spans.append((start, end))
            if tag not in hit_tags:
                hit_tags.append(tag)
        return hit_tags

    @staticmethod
    def _extract_meal(diet_intent: bool, meal_hit: Optional[tuple[int, str]], notices: list[str]) -> Optional[MealTime]:
        """餐次提取：多餐次命中取出现位置最早者；无命中缺省午餐（静默，不追加提示）。"""
        if not diet_intent:
            return None
        if meal_hit is None:
            return MealTime.LUNCH
        return MEAL_KEYWORDS[meal_hit[1]]

    @staticmethod
    def _extract_scene(exercise_intent: bool, text: str, notices: list[str]) -> Optional[Scene]:
        """场景提取：无命中缺省操场 + 提示。"""
        if not exercise_intent:
            return None
        scene_hits = [
            (text.find(keyword), keyword) for keyword in SCENE_KEYWORDS if keyword in text
        ]
        if not scene_hits:
            notices.append(SCENE_DEFAULT_NOTICE)
            return Scene.PLAYGROUND
        return SCENE_KEYWORDS[min(scene_hits)[1]]

    @staticmethod
    def _extract_minutes(exercise_intent: bool, text: str) -> tuple[Optional[int], bool]:
        """时长提取：首个"数字+分钟"匹配；缺省静默 15（无提示）。

        返回 (时长, 是否缺省)：缺省路径（无匹配或超 1-240 范围）置 True，
        使"缺省 15"与"显式 15"对路由层可区分。
        """
        if not exercise_intent:
            return None, False
        match = DURATION_PATTERN.search(text)
        if match:
            minutes = int(match.group(1))
            if MIN_MINUTES <= minutes <= MAX_MINUTES:
                return minutes, False
        return DEFAULT_MINUTES, True

    @staticmethod
    def _extract_weather(exercise_intent: bool, text: str) -> Optional[str]:
        """天气提取：无命中不携带（不触发跑步课表天气降级）。"""
        if not exercise_intent:
            return None
        for keyword, weather in WEATHER_KEYWORDS.items():
            if keyword in text:
                return weather
        return None