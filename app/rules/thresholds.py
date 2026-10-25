"""医学准则阈值常量：物化自《医学与膳食规则库》知识库。

阈值与判定逻辑分离维护（design.md 决策 D1）：更新阈值只改本文件，
不改 app/rules/medical.py 的判定逻辑；与知识库的偏差视为缺陷（spec 4.2.3）。
"""

from pydantic import BaseModel, Field


class UricAcidThresholds(BaseModel):
    """血尿酸阈值（μmol/L），按性别区分。"""

    male: float = 420.0
    female: float = 360.0


class LipidThresholds(BaseModel):
    """血脂四分项阈值（mmol/L）。"""

    total_cholesterol_high: float = 5.2
    triglyceride_high: float = 1.7
    ldl_c_high: float = 3.4
    hdl_c_low: float = 1.0


class GlucoseThresholds(BaseModel):
    """空腹血糖分级阈值（mmol/L）。"""

    high_risk: float = 7.0
    impaired_low: float = 6.1


class BloodPressureThresholds(BaseModel):
    """血压分级阈值（mmHg）。"""

    high_systolic: float = 140.0
    high_diastolic: float = 90.0
    elevated_systolic: float = 120.0
    elevated_diastolic: float = 80.0


class BmiThresholds(BaseModel):
    """BMI 分级阈值（kg/m²）。"""

    underweight: float = 18.5
    overweight: float = 24.0
    obese: float = 28.0


class HemoglobinThresholds(BaseModel):
    """血红蛋白贫血阈值（g/L），按性别区分。"""

    male: float = 120.0
    female: float = 110.0


class MedicalThresholds(BaseModel):
    """医学准则阈值常量聚合（冻结只读，进程启动期创建）。"""

    model_config = {"frozen": True}

    uric_acid: UricAcidThresholds = Field(default_factory=UricAcidThresholds)
    lipid: LipidThresholds = Field(default_factory=LipidThresholds)
    fasting_glucose: GlucoseThresholds = Field(default_factory=GlucoseThresholds)
    blood_pressure: BloodPressureThresholds = Field(default_factory=BloodPressureThresholds)
    bmi: BmiThresholds = Field(default_factory=BmiThresholds)
    hemoglobin: HemoglobinThresholds = Field(default_factory=HemoglobinThresholds)
    h_pylori_positive: str = "阳性"


MEDICAL_THRESHOLDS = MedicalThresholds()

METRIC_RANGES: dict[str, tuple[float, float]] = {
    "uric_acid": (90.0, 1000.0),
    "total_cholesterol": (2.0, 20.0),
    "triglyceride": (0.3, 20.0),
    "ldl_c": (1.0, 10.0),
    "hdl_c": (0.5, 5.0),
    "fasting_glucose": (2.0, 35.0),
    "systolic_bp": (60.0, 260.0),
    "diastolic_bp": (30.0, 180.0),
    "hemoglobin": (50.0, 200.0),
}