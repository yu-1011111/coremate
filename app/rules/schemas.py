"""健康数据契约：以 Pydantic v2 定义规则引擎的全部领域对象。

字段命名以需求规格说明书（spec.md）第 6 章为冻结契约，禁止重命名。
"""

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field

DISCLAIMER_TEXT = "本建议仅供日常健康管理参考，不具备临床诊断效力"

MISSING_MESSAGE = "部分指标未提供，未参与判定"
OUT_OF_RANGE_MESSAGE = "数值异常，请核对体检报告"
GENDER_REQUIRED_MESSAGE = "请补充性别信息以完成该项判定"
HEIGHT_WEIGHT_INCOMPLETE_MESSAGE = "BMI 判定需同时提供身高与体重"
UNKNOWN_FIELD_NOTICE = "存在未识别的指标字段，已被忽略"


class SkipReason(str, Enum):
    """指标未参与判定的原因分类。"""

    MISSING = "MISSING"
    OUT_OF_RANGE = "OUT_OF_RANGE"
    GENDER_REQUIRED = "GENDER_REQUIRED"
    HEIGHT_WEIGHT_INCOMPLETE = "HEIGHT_WEIGHT_INCOMPLETE"


class UserProfile(BaseModel):
    """用户基本档案。gender 必填（尿酸、贫血判定依赖性别阈值）。"""

    gender: Literal["男", "女"]
    age: Optional[int] = Field(default=None, ge=10, le=100, description="年龄（10-100）")
    height: Optional[float] = Field(default=None, ge=100, le=250, description="身高，单位 cm（100-250）")
    weight: Optional[float] = Field(default=None, ge=20, le=300, description="体重，单位 kg（20-300）")


class HealthMetrics(BaseModel):
    """体检指标：10 项字段全部选填，缺失字段由引擎记入 skipped_metrics。

    数值字段的生理合理范围校验由引擎按 METRIC_RANGES 逐项执行：
    超范围时跳过该指标而不拒绝整请求（spec 5.1.3.2 业务容错）。
    extra="allow" 用于识别未知字段并提示，判定时天然忽略。
    """

    model_config = {"extra": "allow"}

    uric_acid: Optional[float] = Field(default=None, description="血尿酸 μmol/L（合理范围 90-1000）")
    total_cholesterol: Optional[float] = Field(default=None, description="总胆固醇 mmol/L（合理范围 2.0-20.0）")
    triglyceride: Optional[float] = Field(default=None, description="甘油三酯 mmol/L（合理范围 0.3-20.0）")
    ldl_c: Optional[float] = Field(default=None, description="低密度脂蛋白 mmol/L（合理范围 1.0-10.0）")
    hdl_c: Optional[float] = Field(default=None, description="高密度脂蛋白 mmol/L（合理范围 0.5-5.0）")
    fasting_glucose: Optional[float] = Field(default=None, description="空腹血糖 mmol/L（合理范围 2.0-35.0）")
    systolic_bp: Optional[float] = Field(default=None, description="收缩压 mmHg（合理范围 60-260）")
    diastolic_bp: Optional[float] = Field(default=None, description="舒张压 mmHg（合理范围 30-180）")
    hemoglobin: Optional[float] = Field(default=None, description="血红蛋白 g/L（合理范围 50-200）")
    h_pylori: Optional[Literal["阳性", "阴性"]] = Field(default=None, description="幽门螺杆菌检测结果")


class HealthTag(BaseModel):
    """健康标签：规则引擎对超标指标生成的结构化结论。"""

    code: str = Field(description="标签编码，全局唯一且一经发布禁止变更")
    name: str = Field(description="中文展示名")
    diet_taboos: list[str] = Field(min_length=1, description="关联饮食禁忌条目")
    exercise_taboos: list[str] = Field(min_length=1, description="关联运动禁忌条目")
    source_metric: str = Field(description="触发该标签的指标字段名")


class SkippedMetric(BaseModel):
    """未参与判定的指标及原因（机器可读 reason + 人类可读 message）。"""

    metric: str = Field(description="指标字段名")
    reason: SkipReason = Field(description="未判定原因")
    message: str = Field(description="面向用户的中文提示文本")


class AnalysisResult(BaseModel):
    """健康分析结果：标签 + 合并去重禁忌 + 强制免责声明。"""

    tags: list[HealthTag] = Field(default_factory=list, description="触发的健康标签列表")
    diet_taboos: list[str] = Field(default_factory=list, description="合并去重后的饮食禁忌")
    exercise_taboos: list[str] = Field(default_factory=list, description="合并去重后的运动禁忌")
    disclaimer: str = Field(default=DISCLAIMER_TEXT, description="免责声明（固定文本，禁止改写）")
    skipped_metrics: list[SkippedMetric] = Field(default_factory=list, description="未参与判定的指标及原因")
    notices: list[str] = Field(default_factory=list, description="其他提示信息")


class AnalysisRequest(BaseModel):
    """健康分析请求聚合模型。"""

    profile: UserProfile
    metrics: HealthMetrics