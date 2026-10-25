"""心理数据目录：解析五张心理知识库表为只读数据资产（设计决策 D2/D3/D6/D10）。

职责：load_records 解析 → 同构唯一性校验 → 多值字段拆分 → ref/link
跨库引用识别与 K 前缀歧义消解 → 引用存在性校验（含音频文件探测）→
触发信号派生字段解析 → campus 电话保真与 verify_date 时效校验 →
进程级缓存固化。import 方向 mood_loader → rec_catalog 单向依赖无循环。
"""

import logging
import re
from datetime import date
from pathlib import Path
from typing import Optional

from app.data.loader import load_records
from app.data.mood_tables import (
    CAMPUS_RESOURCES_TABLE_TEXT,
    MOOD_KB_TABLE_TEXT,
    MOOD_RESOURCES_TABLE_TEXT,
    MOOD_SELFCHECK_TABLE_TEXT,
    MOOD_SIGNALS_TABLE_TEXT,
)
from app.data.rec_catalog import create_default_catalog
from app.rules.mood_schemas import (
    CampusResourceRecord,
    CrossRef,
    MoodKbRecord,
    MoodResourceRecord,
    MoodSignalRecord,
    RefType,
    SelfcheckItemRecord,
)

logger = logging.getLogger("coremate.data")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
AUDIO_DIR = _PROJECT_ROOT / "static" / "audio"

_KB_ID_PATTERN = re.compile(r"(?<![0-9A-Za-z])K(\d{2})(?![0-9])")
_DRINK_ID_PATTERN = re.compile(r"(?<![0-9A-Za-z])K(\d{3})(?![0-9])")
_ACTION_ID_PATTERN = re.compile(r"(?<![0-9A-Za-z])A(\d{3})(?![0-9])")
_RUN_ID_PATTERN = re.compile(r"(?<![0-9A-Za-z])R(\d{3})(?![0-9])")
_DISH_ID_PATTERN = re.compile(r"(?<![0-9A-Za-z])L(\d{2})(?![0-9])")
_CAMPUS_ID_PATTERN = re.compile(r"(?<![0-9A-Za-z])C(\d{2})(?![0-9])")
_CONSECUTIVE_DAYS_PATTERN = re.compile(r"连续\s*(\d+)\s*天")
_SC_ID_PATTERN = re.compile(r"(?<![0-9A-Za-z])SC(\d{1,2})(?![0-9])")

DEFAULT_CONSECUTIVE_DAYS = 3


def _split_comma_tags(value: Optional[str]) -> frozenset[str]:
    """按逗号拆分多值字段（mood / kb_ids / referral_id）。"""
    if not value:
        return frozenset()
    parts = (part.strip() for part in value.split(","))
    return frozenset(part for part in parts if part)


def _split_pipe_tags(value: Optional[str]) -> frozenset[str]:
    """按字面 `\\|`（反斜杠+竖线）拆分多值字段（keywords）。"""
    if not value:
        return frozenset()
    parts = (part.strip() for part in value.split("\\|"))
    return frozenset(part for part in parts if part)


def _split_link_segments(value: Optional[str]) -> list[str]:
    """按"全角｜、半角|、表格转义 `\\|`"三种分隔符统一拆分 link 字段。"""
    if not value:
        return []
    normalized = value.replace("\\|", "|")
    parts = (part.strip() for part in re.split(r"[|｜]", normalized))
    return [part for part in parts if part]


def _to_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _dedupe_by_id(items: list, id_getter, table_name: str) -> list:
    """表内 ID 唯一性校验：重复条目拦截并保留首条（同构 rec_catalog）。"""
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


def _parse_cross_ref(segment: str) -> CrossRef:
    """识别单个引用片段的类型与对象标识（设计决策 D3：长前缀/三位数优先）。

    顺序即优先级：audio（.mp3）→ drink（K 三位数）→ mood_kb（K 两位数）→
    action（A 三位数）→ run_course（R 三位数）→ dish（L 两位数）→
    campus（C 两位数）→ script（话术，无编号模式）。
    K 三位数先于两位数判定，消解 K 前缀歧义。
    """
    if ".mp3" in segment.lower():
        filename = segment.strip()
        available = (AUDIO_DIR / filename).is_file()
        return CrossRef(ref_type=RefType.AUDIO, object_id=filename, display_text=segment, available=available)

    drink_match = _DRINK_ID_PATTERN.search(segment)
    if drink_match:
        drink_id = f"K{drink_match.group(1)}"
        return CrossRef(ref_type=RefType.DRINK, object_id=drink_id, display_text=segment, available=True)

    kb_match = _KB_ID_PATTERN.search(segment)
    if kb_match:
        kb_id = f"K{kb_match.group(1)}"
        return CrossRef(ref_type=RefType.MOOD_KB, object_id=kb_id, display_text=segment, available=True)

    action_match = _ACTION_ID_PATTERN.search(segment)
    if action_match:
        action_id = f"A{action_match.group(1)}"
        return CrossRef(ref_type=RefType.ACTION, object_id=action_id, display_text=segment, available=True)

    run_match = _RUN_ID_PATTERN.search(segment)
    if run_match:
        run_id = f"R{run_match.group(1)}"
        return CrossRef(ref_type=RefType.RUN_COURSE, object_id=run_id, display_text=segment, available=True)

    dish_match = _DISH_ID_PATTERN.search(segment)
    if dish_match:
        dish_id = f"L{dish_match.group(1)}"
        return CrossRef(ref_type=RefType.FOOD, object_id=dish_id, display_text=segment, available=True)

    campus_match = _CAMPUS_ID_PATTERN.search(segment)
    if campus_match:
        campus_id = f"C{campus_match.group(1)}"
        return CrossRef(ref_type=RefType.CAMPUS, object_id=campus_id, display_text=segment, available=True)

    return CrossRef(ref_type=RefType.SCRIPT, object_id=None, display_text=segment, available=True)


def _verify_cross_ref(
    ref: CrossRef,
    kb_ids: frozenset[str],
    campus_ids: frozenset[str],
    rec_catalog,
) -> CrossRef:
    """引用存在性校验：不存在的编号降级 available=False；音频探测文件存在性。"""
    if ref.ref_type is RefType.AUDIO:
        if not ref.available:
            logger.warning("音频文件缺失，引用降级: %s", ref.object_id)
        return ref
    if ref.ref_type is RefType.SCRIPT:
        return ref
    if ref.ref_type is RefType.MOOD_KB:
        available = ref.object_id in kb_ids
    elif ref.ref_type is RefType.CAMPUS:
        available = ref.object_id in campus_ids
    elif ref.ref_type is RefType.ACTION:
        available = any(action.action_id == ref.object_id for action in rec_catalog.actions)
    elif ref.ref_type is RefType.RUN_COURSE:
        available = any(course.course_id == ref.object_id for course in rec_catalog.run_courses)
    elif ref.ref_type is RefType.DRINK:
        available = any(drink.drink_id == ref.object_id for drink in rec_catalog.drinks)
    elif ref.ref_type is RefType.FOOD:
        available = any(dish.dish_id == ref.object_id for dish in rec_catalog.dishes)
    else:
        available = False
    if not available:
        logger.warning("跨库引用对象不存在，降级: %s %s", ref.ref_type.value, ref.object_id)
    return ref.model_copy(update={"available": available})


def _parse_signal_derived(record: dict) -> tuple[int, frozenset[str]]:
    """从 trigger_rule 解析派生字段（决策 D6，数据驱动零改码）。

    行为信号：提取"连续 N 天"作为 min_consecutive_days（无该模式按默认 3）。
    自报信号：提取 SC 编号集合生成 sc_ids。
    """
    rule_text = record.get("trigger_rule") or ""
    days_match = _CONSECUTIVE_DAYS_PATTERN.search(rule_text)
    if days_match:
        min_days = int(days_match.group(1))
    else:
        min_days = DEFAULT_CONSECUTIVE_DAYS
        if "天" in rule_text and "连续" in rule_text:
            logger.warning("行为信号无法解析连续天数，按默认 %d 天处理: %s", DEFAULT_CONSECUTIVE_DAYS, record.get("signal_id"))
    sc_ids = frozenset(f"SC{m}" for m in _SC_ID_PATTERN.findall(rule_text))
    return min_days, sc_ids


def _check_verify_date(record: CampusResourceRecord) -> None:
    """verify_date 时效校验：距当前日期超过 12 个月 → 告警提醒复核（spec 4.4.5）。"""
    if not record.verify_date:
        return
    try:
        year, month = (int(part) for part in record.verify_date.split("-"))
        verified = date(year, month, 1)
    except ValueError:
        logger.warning("verify_date 解析失败: %s %s", record.res_id, record.verify_date)
        return
    today = date.today()
    months_elapsed = (today.year - verified.year) * 12 + (today.month - verified.month)
    if months_elapsed > 12:
        logger.warning("求助资源 %s 校验时间距今超过 12 个月（%s），请复核电话", record.res_id, record.verify_date)


class MoodCatalog:
    """心理域只读数据资产：五类记录的有序元组，含解析完成的 CrossRef 与派生字段。"""

    def __init__(
        self,
        kb_entries: list[MoodKbRecord],
        selfcheck_items: list[SelfcheckItemRecord],
        signals: list[MoodSignalRecord],
        mood_resources: list[MoodResourceRecord],
        campus_resources: list[CampusResourceRecord],
    ) -> None:
        self._kb_entries = tuple(kb_entries)
        self._selfcheck_items = tuple(selfcheck_items)
        self._signals = tuple(signals)
        self._mood_resources = tuple(mood_resources)
        self._campus_resources = tuple(campus_resources)
        if not self._kb_entries:
            logger.warning("心理手册条目数为 0，手册关联将不可用")
        if not self._selfcheck_items:
            logger.warning("便签自查表条目数为 0，自查引擎将返回数据暂不可用提示")
        if not self._signals:
            logger.warning("触发信号条目数为 0，信号引擎将返回数据暂不可用提示")
        if not self._mood_resources:
            logger.warning("资源列表条目数为 0，档位资源将不可用")
        if not self._campus_resources:
            logger.warning("校园求助资源条目数为 0，危机链路资源将不完整")

    @property
    def kb_entries(self) -> tuple[MoodKbRecord, ...]:
        return self._kb_entries

    @property
    def selfcheck_items(self) -> tuple[SelfcheckItemRecord, ...]:
        return self._selfcheck_items

    @property
    def signals(self) -> tuple[MoodSignalRecord, ...]:
        return self._signals

    @property
    def mood_resources(self) -> tuple[MoodResourceRecord, ...]:
        return self._mood_resources

    @property
    def campus_resources(self) -> tuple[CampusResourceRecord, ...]:
        return self._campus_resources

    def get_kb(self, kb_id: str) -> Optional[MoodKbRecord]:
        for entry in self._kb_entries:
            if entry.kb_id == kb_id:
                return entry
        return None

    def get_campus(self, res_id: str) -> Optional[CampusResourceRecord]:
        for record in self._campus_resources:
            if record.res_id == res_id:
                return record
        return None

    def get_signal(self, signal_id: str) -> Optional[MoodSignalRecord]:
        for record in self._signals:
            if record.signal_id == signal_id:
                return record
        return None

    def get_selfcheck(self, sc_id: str) -> Optional[SelfcheckItemRecord]:
        for record in self._selfcheck_items:
            if record.sc_id == sc_id:
                return record
        return None


_DEFAULT_CATALOG: Optional[MoodCatalog] = None


def _build_mood_catalog() -> MoodCatalog:
    rec_catalog = create_default_catalog()

    campus_records = _dedupe_by_id(
        [
            CampusResourceRecord(
                res_id=r["res_id"] or "",
                name=r["name"] or "",
                rtype=r.get("rtype"),
                phone=(r.get("phone") if r.get("phone") and r.get("phone") != "—" else None),
                hours=r.get("hours"),
                location=(r.get("location") if r.get("location") and r.get("location") != "—" else None),
                campus=r.get("campus"),
                note=r.get("note"),
                verify_date=r.get("verify_date"),
            )
            for r in load_records(CAMPUS_RESOURCES_TABLE_TEXT)
        ],
        lambda c: c.res_id,
        "校园求助资源表",
    )
    campus_ids = frozenset(c.res_id for c in campus_records)
    for record in campus_records:
        _check_verify_date(record)

    kb_ids = frozenset(r["kb_id"] for r in load_records(MOOD_KB_TABLE_TEXT))

    kb_records = _dedupe_by_id(
        [
            MoodKbRecord(
                kb_id=r["kb_id"] or "",
                chapter=r.get("chapter"),
                page=r.get("page"),
                mood=_split_comma_tags(r.get("mood")),
                problem=r.get("problem"),
                method=r.get("method"),
                howto=r.get("howto"),
                note=r.get("note"),
                links=tuple(
                    _verify_cross_ref(_parse_cross_ref(segment), kb_ids, campus_ids, rec_catalog)
                    for segment in _split_link_segments(r.get("link"))
                ),
            )
            for r in load_records(MOOD_KB_TABLE_TEXT)
        ],
        lambda k: k.kb_id,
        "心理手册表",
    )

    selfcheck_records = _dedupe_by_id(
        [
            SelfcheckItemRecord(
                sc_id=r["sc_id"] or "",
                text_original=r.get("text_original"),
                mood_level=r.get("mood_level") or "M1",
                kb_ids=_split_comma_tags(r.get("kb_ids")),
                has_referral=(r.get("has_referral") or "0").strip() == "1",
                referral_ids=_split_comma_tags(r.get("referral_id")),
            )
            for r in load_records(MOOD_SELFCHECK_TABLE_TEXT)
        ],
        lambda s: s.sc_id,
        "便签自查表",
    )

    signal_records = _dedupe_by_id(
        [
            MoodSignalRecord(
                signal_id=r["signal_id"] or "",
                category=r.get("category"),
                trigger_rule=r.get("trigger_rule"),
                keywords=_split_pipe_tags(r.get("keywords")),
                target_mood=r.get("target_mood"),
                weight=_to_int(r.get("weight")) or 0,
                min_consecutive_days=min_days,
                sc_ids=sc_ids,
            )
            for r, (min_days, sc_ids) in (
                (r, _parse_signal_derived(r)) for r in load_records(MOOD_SIGNALS_TABLE_TEXT)
            )
        ],
        lambda s: s.signal_id,
        "触发信号表",
    )

    resource_records = _dedupe_by_id(
        [
            MoodResourceRecord(
                res_id=r["res_id"] or "",
                mood_level=r.get("mood_level") or "",
                res_type=r.get("res_type"),
                ref=(
                    _verify_cross_ref(_parse_cross_ref(r.get("ref") or ""), kb_ids, campus_ids, rec_catalog)
                    if r.get("ref")
                    else None
                ),
                name=r.get("name"),
                minutes=_to_int(r.get("minutes")),
                where=r.get("where"),
                mechanism=r.get("mechanism"),
                priority=_to_int(r.get("priority")) or 0,
            )
            for r in load_records(MOOD_RESOURCES_TABLE_TEXT)
        ],
        lambda m: m.res_id,
        "资源列表表",
    )

    return MoodCatalog(
        kb_entries=kb_records,
        selfcheck_items=selfcheck_records,
        signals=signal_records,
        mood_resources=resource_records,
        campus_resources=campus_records,
    )


def create_default_mood_catalog() -> MoodCatalog:
    """进程级缓存工厂：同一进程仅构建一次且全程内容不变（spec 5.1.1.6）。"""
    global _DEFAULT_CATALOG
    if _DEFAULT_CATALOG is None:
        _DEFAULT_CATALOG = _build_mood_catalog()
    return _DEFAULT_CATALOG