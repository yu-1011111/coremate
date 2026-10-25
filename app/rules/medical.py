"""医学规则引擎核心：判定器注册表 + 分析编排。

纯本地计算：不调用大模型、不发起网络 IO（design.md 决策 D6）。
新增一类健康指标只需实现 MetricRule 协议并注册进 DEFAULT_RULES，
analyze 编排逻辑零改动（design.md 决策 D2，spec 4.4.3）。
"""

import logging
from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

from app.data.taboo_catalog import TabooCatalog, TabooBundle, TagDefinition
from app.rules.schemas import (
    DISCLAIMER_TEXT,
    GENDER_REQUIRED_MESSAGE,
    HEIGHT_WEIGHT_INCOMPLETE_MESSAGE,
    MISSING_MESSAGE,
    UNKNOWN_FIELD_NOTICE,
    AnalysisResult,
    HealthMetrics,
    HealthTag,
    SkippedMetric,
    SkipReason,
    UserProfile,
)
from app.rules.thresholds import MEDICAL_THRESHOLDS, METRIC_RANGES, MedicalThresholds

audit_logger = logging.getLogger("coremate.audit")
audit_logger.propagate = False


@dataclass
class EvaluationContext:
    """判定上下文：打包档案、指标、阈值常量与禁忌目录。"""

    profile: UserProfile
    metrics: HealthMetrics
    thresholds: MedicalThresholds
    catalog: TabooCatalog


@dataclass
class RuleOutcome:
    """单判定器产出：触发标签列表 + 未判定提示列表。"""

    tags: list[HealthTag] = field(default_factory=list)
    skipped: list[SkippedMetric] = field(default_factory=list)


@runtime_checkable
class MetricRule(Protocol):
    """判定器协议：每类健康指标一个实现。"""

    metric_fields: tuple[str, ...]

    def evaluate(self, ctx: EvaluationContext) -> RuleOutcome: ...


def _missing(metric: str) -> SkippedMetric:
    return SkippedMetric(metric=metric, reason=SkipReason.MISSING, message=MISSING_MESSAGE)


def _out_of_range(metric: str) -> SkippedMetric:
    return SkippedMetric(metric=metric, reason=SkipReason.OUT_OF_RANGE, message="数值异常，请核对体检报告")


def _gender_required(metric: str) -> SkippedMetric:
    return SkippedMetric(metric=metric, reason=SkipReason.GENDER_REQUIRED, message=GENDER_REQUIRED_MESSAGE)


def _height_weight_incomplete() -> SkippedMetric:
    return SkippedMetric(
        metric="bmi",
        reason=SkipReason.HEIGHT_WEIGHT_INCOMPLETE,
        message=HEIGHT_WEIGHT_INCOMPLETE_MESSAGE,
    )


def _value_in_range(metric: str, value: float) -> bool:
    low, high = METRIC_RANGES[metric]
    return low <= value <= high


def _make_tag(ctx: EvaluationContext, code: str, source_metric: str) -> HealthTag:
    definition: TagDefinition | None = ctx.catalog.get_definition(code)
    if definition is None:
        raise ValueError(f"标签编码未在禁忌目录登记: {code}")
    return HealthTag(
        code=code,
        name=definition.name,
        diet_taboos=list(definition.diet_taboos),
        exercise_taboos=list(definition.exercise_taboos),
        source_metric=source_metric,
    )


class UricAcidRule:
    """尿酸判定：男 > 420 / 女 > 360 μmol/L 触发"高尿酸"。"""

    metric_fields = ("uric_acid",)

    def evaluate(self, ctx: EvaluationContext) -> RuleOutcome:
        outcome = RuleOutcome()
        value = ctx.metrics.uric_acid
        if value is None:
            outcome.skipped.append(_missing("uric_acid"))
            return outcome
        if not _value_in_range("uric_acid", value):
            outcome.skipped.append(_out_of_range("uric_acid"))
            return outcome
        gender = ctx.profile.gender
        if gender not in ("男", "女"):
            outcome.skipped.append(_gender_required("uric_acid"))
            return outcome
        limit = ctx.thresholds.uric_acid.male if gender == "男" else ctx.thresholds.uric_acid.female
        if value > limit:
            outcome.tags.append(_make_tag(ctx, "hyperuricemia", "uric_acid"))
        return outcome


class LipidPanelRule:
    """血脂判定：四分项独立判定，可同时触发多个标签。"""

    metric_fields = ("total_cholesterol", "triglyceride", "ldl_c", "hdl_c")

    def evaluate(self, ctx: EvaluationContext) -> RuleOutcome:
        outcome = RuleOutcome()
        checks = (
            ("total_cholesterol", "hypercholesterolemia", lambda v, t: v >= t.lipid.total_cholesterol_high),
            ("triglyceride", "hypertriglyceridemia", lambda v, t: v >= t.lipid.triglyceride_high),
            ("ldl_c", "high_ldl_c", lambda v, t: v >= t.lipid.ldl_c_high),
            ("hdl_c", "low_hdl_c", lambda v, t: v < t.lipid.hdl_c_low),
        )
        for metric, code, predicate in checks:
            value = getattr(ctx.metrics, metric)
            if value is None:
                outcome.skipped.append(_missing(metric))
                continue
            if not _value_in_range(metric, value):
                outcome.skipped.append(_out_of_range(metric))
                continue
            if predicate(value, ctx.thresholds):
                outcome.tags.append(_make_tag(ctx, code, metric))
        return outcome


class FastingGlucoseRule:
    """空腹血糖判定：分级取最高级，同一请求只输出较高一级标签。"""

    metric_fields = ("fasting_glucose",)

    def evaluate(self, ctx: EvaluationContext) -> RuleOutcome:
        outcome = RuleOutcome()
        value = ctx.metrics.fasting_glucose
        if value is None:
            outcome.skipped.append(_missing("fasting_glucose"))
            return outcome
        if not _value_in_range("fasting_glucose", value):
            outcome.skipped.append(_out_of_range("fasting_glucose"))
            return outcome
        thresholds = ctx.thresholds.fasting_glucose
        if value >= thresholds.high_risk:
            outcome.tags.append(_make_tag(ctx, "hyperglycemia_risk", "fasting_glucose"))
        elif value >= thresholds.impaired_low:
            outcome.tags.append(_make_tag(ctx, "impaired_fasting_glucose", "fasting_glucose"))
        return outcome


class BloodPressureRule:
    """血压判定：收缩压/舒张压分别定级后取较高等级。"""

    metric_fields = ("systolic_bp", "diastolic_bp")

    def evaluate(self, ctx: EvaluationContext) -> RuleOutcome:
        outcome = RuleOutcome()
        readings: dict[str, float] = {}
        for name, value in (("systolic_bp", ctx.metrics.systolic_bp), ("diastolic_bp", ctx.metrics.diastolic_bp)):
            if value is None:
                outcome.skipped.append(_missing(name))
            elif not _value_in_range(name, value):
                outcome.skipped.append(_out_of_range(name))
            else:
                readings[name] = value
        if not readings:
            return outcome

        thresholds = ctx.thresholds.blood_pressure
        sbp = readings.get("systolic_bp")
        dbp = readings.get("diastolic_bp")
        if (sbp is not None and sbp >= thresholds.high_systolic) or (
            dbp is not None and dbp >= thresholds.high_diastolic
        ):
            code = "hypertension"
        elif (sbp is not None and sbp >= thresholds.elevated_systolic) or (
            dbp is not None and dbp >= thresholds.elevated_diastolic
        ):
            code = "elevated_blood_pressure"
        else:
            return outcome
        source_metric = "+".join(readings)
        outcome.tags.append(_make_tag(ctx, code, source_metric))
        return outcome


class BmiRule:
    """BMI 判定：由身高体重计算（不接受直接录入）。

    < 18.5 偏瘦；24 ≤ BMI < 28 超重；≥ 28 肥胖。
    """

    metric_fields = ("height", "weight")

    def evaluate(self, ctx: EvaluationContext) -> RuleOutcome:
        outcome = RuleOutcome()
        height = ctx.profile.height
        weight = ctx.profile.weight
        if height is None and weight is None:
            outcome.skipped.append(_missing("bmi"))
            return outcome
        if height is None or weight is None:
            outcome.skipped.append(_height_weight_incomplete())
            return outcome
        bmi = weight / (height / 100.0) ** 2
        thresholds = ctx.thresholds.bmi
        if bmi < thresholds.underweight:
            code = "underweight"
        elif bmi < thresholds.overweight:
            return outcome
        elif bmi < thresholds.obese:
            code = "overweight"
        else:
            code = "obesity"
        outcome.tags.append(_make_tag(ctx, code, "height+weight"))
        return outcome


class HemoglobinRule:
    """贫血判定：男血红蛋白 < 120 / 女 < 110 g/L 触发"贫血风险"。"""

    metric_fields = ("hemoglobin",)

    def evaluate(self, ctx: EvaluationContext) -> RuleOutcome:
        outcome = RuleOutcome()
        value = ctx.metrics.hemoglobin
        if value is None:
            outcome.skipped.append(_missing("hemoglobin"))
            return outcome
        if not _value_in_range("hemoglobin", value):
            outcome.skipped.append(_out_of_range("hemoglobin"))
            return outcome
        gender = ctx.profile.gender
        if gender not in ("男", "女"):
            outcome.skipped.append(_gender_required("hemoglobin"))
            return outcome
        limit = ctx.thresholds.hemoglobin.male if gender == "男" else ctx.thresholds.hemoglobin.female
        if value < limit:
            outcome.tags.append(_make_tag(ctx, "anemia_risk", "hemoglobin"))
        return outcome


class HPyloriRule:
    """幽门螺杆菌判定：检测结果为"阳性"触发标签。"""

    metric_fields = ("h_pylori",)

    def evaluate(self, ctx: EvaluationContext) -> RuleOutcome:
        outcome = RuleOutcome()
        value = ctx.metrics.h_pylori
        if value is None:
            outcome.skipped.append(_missing("h_pylori"))
            return outcome
        if value == ctx.thresholds.h_pylori_positive:
            outcome.tags.append(_make_tag(ctx, "h_pylori_positive", "h_pylori"))
        return outcome


DEFAULT_RULES: tuple[MetricRule, ...] = (
    UricAcidRule(),
    LipidPanelRule(),
    FastingGlucoseRule(),
    BloodPressureRule(),
    BmiRule(),
    HemoglobinRule(),
    HPyloriRule(),
)


class MedicalRulesEngine:
    """医学规则引擎：无状态、线程安全，可被并发共享（design.md 决策 D6）。"""

    def __init__(
        self,
        rules: Sequence[MetricRule],
        catalog: TabooCatalog,
        thresholds: MedicalThresholds = MEDICAL_THRESHOLDS,
    ) -> None:
        self._rules: tuple[MetricRule, ...] = tuple(rules)
        self._catalog = catalog
        self._thresholds = thresholds

    def analyze(self, profile: UserProfile, metrics: HealthMetrics) -> AnalysisResult:
        """执行健康分析：逐项判定 → 汇总标签 → 合并禁忌 → 注入免责声明。

        单一指标解析失败不影响其余指标判定（spec 4.2.2）；
        相同输入产出完全相同的输出（spec 4.2.1）。
        """
        tags: list[HealthTag] = []
        skipped: list[SkippedMetric] = []
        notices: list[str] = []

        unknown_fields = sorted(metrics.model_extra or {})
        if unknown_fields:
            notices.append(UNKNOWN_FIELD_NOTICE)
            audit_logger.warning("存在未识别的指标字段，已忽略: %s", unknown_fields)

        ctx = EvaluationContext(profile=profile, metrics=metrics, thresholds=self._thresholds, catalog=self._catalog)
        for rule in self._rules:
            outcome = rule.evaluate(ctx)
            tags.extend(outcome.tags)
            skipped.extend(outcome.skipped)

        if any(item.reason is SkipReason.MISSING for item in skipped):
            notices.append(MISSING_MESSAGE)

        bundle: TabooBundle = self._catalog.get_taboos([tag.code for tag in tags])

        result = AnalysisResult(
            tags=tags,
            diet_taboos=bundle.diet_taboos,
            exercise_taboos=bundle.exercise_taboos,
            disclaimer=DISCLAIMER_TEXT,
            skipped_metrics=skipped,
            notices=notices,
        )
        self._audit(profile, metrics, result)
        return result

    def _audit(self, profile: UserProfile, metrics: HealthMetrics, result: AnalysisResult) -> None:
        """审计日志：仅记录字段名摘要与标签编码，不记录指标数值（design.md 决策 D7）。"""
        provided = [name for name in type(metrics).model_fields if getattr(metrics, name) is not None]
        audit_logger.info(
            "健康分析完成 | 性别=%s | 已提供指标=%s | 触发标签=%s | 未判定=%s",
            profile.gender,
            ",".join(provided) or "无",
            ",".join(tag.code for tag in result.tags) or "无",
            ",".join(f"{item.metric}:{item.reason.value}" for item in result.skipped_metrics) or "无",
        )


def create_default_engine() -> MedicalRulesEngine:
    """装配默认 7 个判定器与禁忌目录，返回即用型无状态引擎实例。"""
    return MedicalRulesEngine(rules=DEFAULT_RULES, catalog=TabooCatalog())