"""健康标签到数据表人群标签的映射（独立于引擎逻辑单独维护，spec 4.4.2）。

设计决策 D4：数据表人群标签用词两套并存——菜品库/饮品库对体重维度用
"减脂"，而动作库/跑步课表/体育课库直接用"超重"。为保证 R002-R008 的
block_for 避雷、R001 的 caution_for 谨慎提示与菜品 avoid_for 匹配同时
成立，超重/肥胖映射为双词集合 {减脂, 超重}。

15 标签合法枚举从第一阶段 taboo_catalog 的 TAG_DEFINITIONS 导入，
禁止硬编码第二份清单（避免双源漂移）。
"""

from dataclasses import dataclass
from typing import Sequence

from app.data.taboo_catalog import TAG_DEFINITIONS

KNOWN_HEALTH_TAGS: frozenset[str] = frozenset(definition.name for definition in TAG_DEFINITIONS.values())

HEALTH_TAG_TO_CROWD_TAGS: dict[str, frozenset[str]] = {
    "高尿酸": frozenset({"高尿酸"}),
    "高血糖风险": frozenset({"高血糖", "控糖"}),
    "空腹血糖受损": frozenset({"高血糖", "控糖"}),
    "高血压": frozenset({"高血压"}),
    "血压偏高": frozenset({"高血压"}),
    "高胆固醇": frozenset({"高血脂"}),
    "高甘油三酯": frozenset({"高血脂"}),
    "低密度脂蛋白偏高": frozenset({"高血脂"}),
    "高密度脂蛋白偏低": frozenset({"高血脂"}),
    "超重": frozenset({"减脂", "超重"}),
    "肥胖": frozenset({"减脂", "超重"}),
    "偏瘦": frozenset({"增肌"}),
    "贫血风险": frozenset({"贫血"}),
    "幽门螺杆菌阳性": frozenset({"肠胃敏感"}),
    "肠胃敏感": frozenset({"肠胃敏感"}),
}


@dataclass(frozen=True)
class MappedTags:
    """映射产出：人群标签集合 + 未知标签元组（按输入顺序）。"""

    crowd_tags: frozenset[str]
    unknown_tags: tuple[str, ...]


def map_health_tags(tags: Sequence[str]) -> MappedTags:
    """将健康标签列表映射合并为人群标签集合；体系外标签收集为未知标签。

    纯函数无副作用、不抛异常；空列表返回空集合（健康用户路径，spec 5.2.3.1）。
    """
    crowd: set[str] = set()
    unknown: list[str] = []
    for tag in tags:
        mapped = HEALTH_TAG_TO_CROWD_TAGS.get(tag)
        if mapped is None:
            unknown.append(tag)
            continue
        crowd.update(mapped)
    return MappedTags(crowd_tags=frozenset(crowd), unknown_tags=tuple(unknown))