"""标签-禁忌目录：物化自《医学与膳食规则库》知识库。

维护"健康标签编码 → 饮食禁忌 + 运动禁忌"映射，
与 thresholds.py 同属知识库物化数据，偏差视为缺陷（spec 4.2.3）。
"""

import logging
from typing import Sequence

from pydantic import BaseModel, Field

logger = logging.getLogger("coremate.taboo")


class TagDefinition(BaseModel):
    """健康标签定义：编码、中文展示名与关联禁忌条目。"""

    code: str
    name: str
    diet_taboos: list[str] = Field(min_length=1)
    exercise_taboos: list[str] = Field(min_length=1)


class TabooBundle(BaseModel):
    """多标签禁忌合并结果（已按出现顺序去重）。"""

    diet_taboos: list[str] = Field(default_factory=list)
    exercise_taboos: list[str] = Field(default_factory=list)


TAG_DEFINITIONS: dict[str, TagDefinition] = {
    definition.code: definition
    for definition in (
        TagDefinition(
            code="hyperuricemia",
            name="高尿酸",
            diet_taboos=[
                "限制高嘌呤饮食（如动物内脏、浓肉汤、部分海产品）",
                "避免饮用含果糖饮料与酒精饮品",
            ],
            exercise_taboos=[
                "避免高强度无氧爆发类运动（如短跑冲刺、快速跳绳）",
                "关节不适期应暂停剧烈运动，改为中低强度有氧运动",
            ],
        ),
        TagDefinition(
            code="hypercholesterolemia",
            name="高胆固醇",
            diet_taboos=[
                "减少高胆固醇食物摄入（如动物内脏、蟹黄、鱼籽）",
                "限制饱和脂肪与油炸食品",
            ],
            exercise_taboos=[
                "避免长期静息少动的生活方式，循序渐进安排有氧运动",
            ],
        ),
        TagDefinition(
            code="hypertriglyceridemia",
            name="高甘油三酯",
            diet_taboos=[
                "限制高糖与精制碳水食物（如甜点、含糖饮料）",
                "限制酒精与高脂饮食",
            ],
            exercise_taboos=[
                "避免空腹状态下的高强度运动",
            ],
        ),
        TagDefinition(
            code="high_ldl_c",
            name="低密度脂蛋白偏高",
            diet_taboos=[
                "减少反式脂肪与饱和脂肪摄入（如油炸食品、肥肉、奶油制品）",
            ],
            exercise_taboos=[
                "避免长期高强度静力性训练，以规律有氧运动为主",
            ],
        ),
        TagDefinition(
            code="low_hdl_c",
            name="高密度脂蛋白偏低",
            diet_taboos=[
                "控制精制糖与油炸食品摄入",
            ],
            exercise_taboos=[
                "避免长期久坐，保持规律的中等强度有氧运动",
            ],
        ),
        TagDefinition(
            code="hyperglycemia_risk",
            name="高血糖风险",
            diet_taboos=[
                "限制高糖与高升糖指数食物（如含糖饮料、甜点、精制主食）",
            ],
            exercise_taboos=[
                "避免空腹高强度运动，运动中注意防范低血糖",
            ],
        ),
        TagDefinition(
            code="impaired_fasting_glucose",
            name="空腹血糖受损",
            diet_taboos=[
                "控制精制碳水与甜食摄入，保持规律进餐",
            ],
            exercise_taboos=[
                "避免空腹状态下长时间运动",
            ],
        ),
        TagDefinition(
            code="hypertension",
            name="高血压",
            diet_taboos=[
                "限制钠盐摄入，避免腌制与高盐加工食品",
            ],
            exercise_taboos=[
                "避免憋气发力的大强度力量训练（如大重量深蹲、硬拉）",
            ],
        ),
        TagDefinition(
            code="elevated_blood_pressure",
            name="血压偏高",
            diet_taboos=[
                "减少盐分与高钠加工食品摄入",
            ],
            exercise_taboos=[
                "避免突然发力的短时高强度运动",
            ],
        ),
        TagDefinition(
            code="underweight",
            name="偏瘦",
            diet_taboos=[
                "避免极端节食与低热量饮食，保证充足热量与优质蛋白摄入",
            ],
            exercise_taboos=[
                "避免长时间大消耗量的耐力运动，配合适度力量训练",
            ],
        ),
        TagDefinition(
            code="overweight",
            name="超重",
            diet_taboos=[
                "控制总热量摄入",
                "限制高糖高脂零食与含糖饮料",
            ],
            exercise_taboos=[
                "避免长时间高冲击跑跳运动，注意保护膝踝关节",
            ],
        ),
        TagDefinition(
            code="obesity",
            name="肥胖",
            diet_taboos=[
                "严格控制总热量摄入，限制高糖高脂与精制加工食品",
            ],
            exercise_taboos=[
                "避免高冲击跑跳与负重过大的运动，优先选择游泳、骑行等低冲击项目",
            ],
        ),
        TagDefinition(
            code="anemia_risk",
            name="贫血风险",
            diet_taboos=[
                "避免餐后立即饮用浓茶、咖啡（影响铁吸收），保证铁与蛋白质摄入",
            ],
            exercise_taboos=[
                "避免高强度长时间耐力运动，出现头晕乏力应立即停止",
            ],
        ),
        TagDefinition(
            code="h_pylori_positive",
            name="幽门螺杆菌阳性",
            diet_taboos=[
                "实行分餐制避免交叉感染，避免辛辣刺激与生冷食物",
            ],
            exercise_taboos=[
                "保持规律适度运动即可，无特殊运动禁忌",
            ],
        ),
        TagDefinition(
            code="gastric_sensitive",
            name="肠胃敏感",
            diet_taboos=[
                "避免重度辛辣刺激与生冷食物（如重辣火锅、麻辣烫、刺身、冰饮）",
                "少食多餐，别空腹喝咖啡/浓茶",
            ],
            exercise_taboos=[
                "避免剧烈震荡腹部的核心训练（剧烈卷腹、仰卧起坐），易引起肠胃不适或反流",
            ],
        ),
    )
}


class TabooCatalog:
    """标签禁忌目录：进程级只读数据，提供合并去重查询。"""

    def __init__(self, definitions: dict[str, TagDefinition] | None = None) -> None:
        self._definitions: dict[str, TagDefinition] = dict(
            definitions if definitions is not None else TAG_DEFINITIONS
        )

    def get_definition(self, tag_code: str) -> TagDefinition | None:
        return self._definitions.get(tag_code)

    def get_taboos(self, tag_codes: Sequence[str]) -> TabooBundle:
        """按标签编码集合查询禁忌并合并去重；未登记编码记日志并忽略。"""
        diet: list[str] = []
        exercise: list[str] = []
        for code in tag_codes:
            definition = self._definitions.get(code)
            if definition is None:
                logger.warning("禁忌目录存在未登记的标签编码，已忽略: %s", code)
                continue
            diet.extend(definition.diet_taboos)
            exercise.extend(definition.exercise_taboos)
        return TabooBundle(
            diet_taboos=list(dict.fromkeys(diet)),
            exercise_taboos=list(dict.fromkeys(exercise)),
        )