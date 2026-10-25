"""医学规则引擎测试用例（tasks.md 任务组 7）。

覆盖：七类指标阈值边界、分级取高、禁忌合并与免责声明、
异常容错、确定性与可重放、API 层集成、性能与并发。
"""

import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.rules.medical import create_default_engine
from app.rules.schemas import (
    DISCLAIMER_TEXT,
    MISSING_MESSAGE,
    UNKNOWN_FIELD_NOTICE,
    HealthMetrics,
    SkipReason,
    UserProfile,
)

NORMAL_METRICS = dict(
    uric_acid=300.0,
    total_cholesterol=4.0,
    triglyceride=1.0,
    ldl_c=2.0,
    hdl_c=1.5,
    fasting_glucose=5.0,
    systolic_bp=110.0,
    diastolic_bp=70.0,
    hemoglobin=140.0,
    h_pylori="阴性",
)


@pytest.fixture(scope="session")
def engine():
    return create_default_engine()


@pytest.fixture(scope="session")
def client():
    return TestClient(create_app())


def make_profile(gender="男", age=20, height=170.0, weight=65.0) -> UserProfile:
    return UserProfile(gender=gender, age=age, height=height, weight=weight)


def make_metrics(**overrides) -> HealthMetrics:
    values = dict(
        uric_acid=None,
        total_cholesterol=None,
        triglyceride=None,
        ldl_c=None,
        hdl_c=None,
        fasting_glucose=None,
        systolic_bp=None,
        diastolic_bp=None,
        hemoglobin=None,
        h_pylori=None,
    )
    values.update(overrides)
    return HealthMetrics(**values)


def tag_codes(result) -> list:
    return [tag.code for tag in result.tags]


def skipped_map(result) -> dict:
    return {item.metric: item.reason for item in result.skipped_metrics}


class TestThresholdBoundaries:
    """7.1 七类指标阈值边界测试：正常 / 临界 / 超标三分支。"""

    def test_all_normal_returns_no_tags(self, engine):
        result = engine.analyze(make_profile(), make_metrics(**NORMAL_METRICS))
        assert result.tags == []
        assert result.diet_taboos == []
        assert result.exercise_taboos == []
        assert result.skipped_metrics == []

    @pytest.mark.parametrize(
        "gender,value,expected",
        [
            ("男", 420.0, []),
            ("男", 421.0, ["hyperuricemia"]),
            ("男", 390.0, []),
            ("女", 360.0, []),
            ("女", 361.0, ["hyperuricemia"]),
            ("女", 390.0, ["hyperuricemia"]),
        ],
    )
    def test_uric_acid_gender_boundaries(self, engine, gender, value, expected):
        result = engine.analyze(make_profile(gender=gender), make_metrics(uric_acid=value))
        assert tag_codes(result) == expected

    @pytest.mark.parametrize(
        "metric,value,expected",
        [
            ("total_cholesterol", 5.19, []),
            ("total_cholesterol", 5.2, ["hypercholesterolemia"]),
            ("triglyceride", 1.7, ["hypertriglyceridemia"]),
            ("ldl_c", 3.4, ["high_ldl_c"]),
            ("hdl_c", 1.0, []),
            ("hdl_c", 0.99, ["low_hdl_c"]),
        ],
    )
    def test_lipid_boundaries(self, engine, metric, value, expected):
        result = engine.analyze(make_profile(), make_metrics(**{metric: value}))
        assert tag_codes(result) == expected

    def test_lipid_all_items_triggered_together(self, engine):
        result = engine.analyze(
            make_profile(),
            make_metrics(total_cholesterol=6.0, triglyceride=2.5, ldl_c=4.0, hdl_c=0.8),
        )
        assert tag_codes(result) == [
            "hypercholesterolemia",
            "hypertriglyceridemia",
            "high_ldl_c",
            "low_hdl_c",
        ]

    @pytest.mark.parametrize(
        "value,expected",
        [
            (6.09, []),
            (6.1, ["impaired_fasting_glucose"]),
            (6.99, ["impaired_fasting_glucose"]),
            (7.0, ["hyperglycemia_risk"]),
        ],
    )
    def test_glucose_boundaries(self, engine, value, expected):
        result = engine.analyze(make_profile(), make_metrics(fasting_glucose=value))
        assert tag_codes(result) == expected

    @pytest.mark.parametrize(
        "sbp,dbp,expected",
        [
            (119.0, 79.0, []),
            (120.0, 70.0, ["elevated_blood_pressure"]),
            (119.0, 80.0, ["elevated_blood_pressure"]),
            (139.0, 89.0, ["elevated_blood_pressure"]),
            (140.0, 70.0, ["hypertension"]),
            (139.0, 90.0, ["hypertension"]),
        ],
    )
    def test_blood_pressure_boundaries(self, engine, sbp, dbp, expected):
        result = engine.analyze(make_profile(), make_metrics(systolic_bp=sbp, diastolic_bp=dbp))
        assert tag_codes(result) == expected

    def test_blood_pressure_single_side_grading(self, engine):
        result = engine.analyze(make_profile(), make_metrics(diastolic_bp=85.0))
        assert tag_codes(result) == ["elevated_blood_pressure"]
        assert result.tags[0].source_metric == "diastolic_bp"

    @pytest.mark.parametrize(
        "weight,expected",
        [
            (73.6, ["underweight"]),
            (74.0, []),
            (95.6, []),
            (96.0, ["overweight"]),
            (111.6, ["overweight"]),
            (112.0, ["obesity"]),
        ],
    )
    def test_bmi_boundaries(self, engine, weight, expected):
        result = engine.analyze(make_profile(height=200.0, weight=weight), make_metrics())
        assert tag_codes(result) == expected

    @pytest.mark.parametrize(
        "gender,value,expected",
        [
            ("男", 120.0, []),
            ("男", 119.0, ["anemia_risk"]),
            ("女", 110.0, []),
            ("女", 109.0, ["anemia_risk"]),
        ],
    )
    def test_hemoglobin_gender_boundaries(self, engine, gender, value, expected):
        result = engine.analyze(make_profile(gender=gender), make_metrics(hemoglobin=value))
        assert tag_codes(result) == expected

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("阴性", []),
            ("阳性", ["h_pylori_positive"]),
        ],
    )
    def test_h_pylori_boundaries(self, engine, value, expected):
        result = engine.analyze(make_profile(), make_metrics(h_pylori=value))
        assert tag_codes(result) == expected


class TestHighestGradeOnly:
    """7.2 分级取高测试：同一请求只输出较高一级标签。"""

    def test_glucose_high_risk_excludes_impaired(self, engine):
        result = engine.analyze(make_profile(), make_metrics(fasting_glucose=8.5))
        codes = tag_codes(result)
        assert codes == ["hyperglycemia_risk"]
        assert "impaired_fasting_glucose" not in codes

    def test_blood_pressure_high_grade_takes_precedence(self, engine):
        result = engine.analyze(make_profile(), make_metrics(systolic_bp=145.0, diastolic_bp=85.0))
        codes = tag_codes(result)
        assert codes == ["hypertension"]
        assert "elevated_blood_pressure" not in codes


class TestTabooMergeAndDisclaimer:
    """7.3 禁忌合并去重与免责声明测试。"""

    def test_overweight_scenario_merge_and_disclaimer(self, engine):
        result = engine.analyze(
            make_profile(height=175.0, weight=85.0),
            make_metrics(uric_acid=450.0),
        )
        assert tag_codes(result) == ["hyperuricemia", "overweight"]
        assert result.diet_taboos == [
            "限制高嘌呤饮食（如动物内脏、浓肉汤、部分海产品）",
            "避免饮用含果糖饮料与酒精饮品",
            "控制总热量摄入",
            "限制高糖高脂零食与含糖饮料",
        ]
        assert result.exercise_taboos == [
            "避免高强度无氧爆发类运动（如短跑冲刺、快速跳绳）",
            "关节不适期应暂停剧烈运动，改为中低强度有氧运动",
            "避免长时间高冲击跑跳运动，注意保护膝踝关节",
        ]
        assert result.disclaimer == "本建议仅供日常健康管理参考，不具备临床诊断效力"

    def test_disclaimer_always_present(self, engine):
        result = engine.analyze(make_profile(), make_metrics(uric_acid=300.0))
        assert result.disclaimer == DISCLAIMER_TEXT

    def test_no_tags_means_no_taboos(self, engine):
        result = engine.analyze(make_profile(), make_metrics(**NORMAL_METRICS))
        assert result.diet_taboos == []
        assert result.exercise_taboos == []


class TestFaultTolerance:
    """7.4 异常容错测试：单指标失败不拖垮整体分析。"""

    def test_missing_metrics_are_skipped(self, engine):
        result = engine.analyze(make_profile(), make_metrics(uric_acid=300.0))
        skipped = skipped_map(result)
        assert skipped["total_cholesterol"] is SkipReason.MISSING
        assert skipped["fasting_glucose"] is SkipReason.MISSING
        assert skipped["h_pylori"] is SkipReason.MISSING
        assert "uric_acid" not in skipped
        assert MISSING_MESSAGE in result.notices

    @pytest.mark.parametrize(
        "overrides,bad_metric",
        [
            ({"uric_acid": 5000.0}, "uric_acid"),
            ({"fasting_glucose": -1.0}, "fasting_glucose"),
            ({"systolic_bp": 300.0}, "systolic_bp"),
        ],
    )
    def test_out_of_range_skips_only_that_metric(self, engine, overrides, bad_metric):
        result = engine.analyze(make_profile(), make_metrics(**{**NORMAL_METRICS, **overrides}))
        skipped = skipped_map(result)
        assert skipped[bad_metric] is SkipReason.OUT_OF_RANGE
        assert "总胆固醇" not in "".join(tag.name for tag in result.tags)
        assert tag_codes(result) == []

    def test_out_of_range_message_is_user_friendly(self, engine):
        result = engine.analyze(make_profile(), make_metrics(uric_acid=5000.0))
        item = next(s for s in result.skipped_metrics if s.metric == "uric_acid")
        assert item.message == "数值异常，请核对体检报告"

    def test_gender_missing_suspends_gender_rules(self, engine):
        profile = UserProfile.model_construct(gender=None)
        result = engine.analyze(profile, make_metrics(uric_acid=450.0, hemoglobin=100.0))
        skipped = skipped_map(result)
        assert skipped["uric_acid"] is SkipReason.GENDER_REQUIRED
        assert skipped["hemoglobin"] is SkipReason.GENDER_REQUIRED
        assert tag_codes(result) == []
        for item in result.skipped_metrics:
            if item.reason is SkipReason.GENDER_REQUIRED:
                assert item.message == "请补充性别信息以完成该项判定"

    def test_bmi_requires_both_height_and_weight(self, engine):
        result = engine.analyze(make_profile(height=170.0, weight=None), make_metrics(**NORMAL_METRICS))
        assert skipped_map(result)["bmi"] is SkipReason.HEIGHT_WEIGHT_INCOMPLETE

        result = engine.analyze(make_profile(height=None, weight=65.0), make_metrics(**NORMAL_METRICS))
        assert skipped_map(result)["bmi"] is SkipReason.HEIGHT_WEIGHT_INCOMPLETE

    def test_bmi_missing_when_height_weight_absent(self, engine):
        result = engine.analyze(make_profile(height=None, weight=None), make_metrics(uric_acid=300.0))
        assert skipped_map(result)["bmi"] is SkipReason.MISSING

    def test_unknown_fields_are_ignored(self, engine):
        result = engine.analyze(make_profile(), make_metrics(uric_acid=300.0, unknown_metric="x"))
        assert UNKNOWN_FIELD_NOTICE in result.notices
        assert MISSING_MESSAGE in result.notices
        assert tag_codes(result) == []


class TestDeterminism:
    """7.5 确定性与可重放测试。"""

    def test_same_input_same_output(self, engine):
        profile = make_profile(gender="女", height=160.0, weight=70.0)
        metrics = make_metrics(**{**NORMAL_METRICS, "uric_acid": 380.0, "hemoglobin": 105.0})
        dumps = [engine.analyze(profile, metrics).model_dump() for _ in range(3)]
        assert dumps[0] == dumps[1] == dumps[2]
        assert tag_codes(engine.analyze(profile, metrics)) == [
            "hyperuricemia",
            "overweight",
            "anemia_risk",
        ]


class TestApiIntegration:
    """7.6 API 层集成测试：TestClient 验证 200 与 422 场景。"""

    def test_health_check(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "service": "coremate-medical-rules-engine"}

    def test_analysis_returns_full_result(self, client):
        response = client.post(
            "/api/v1/health/analysis",
            json={
                "profile": {"gender": "男", "age": 20, "height": 175.0, "weight": 85.0},
                "metrics": {"uric_acid": 450.0, "h_pylori": "阴性"},
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert [tag["name"] for tag in body["tags"]] == ["高尿酸", "超重"]
        assert body["disclaimer"] == "本建议仅供日常健康管理参考，不具备临床诊断效力"
        assert body["diet_taboos"]
        assert isinstance(body["skipped_metrics"], list)
        assert isinstance(body["notices"], list)

    def test_analysis_all_normal_via_api(self, client):
        response = client.post(
            "/api/v1/health/analysis",
            json={"profile": {"gender": "女", "height": 165.0, "weight": 55.0}, "metrics": NORMAL_METRICS},
        )
        assert response.status_code == 200
        assert response.json()["tags"] == []

    def test_unknown_metric_field_notice_via_api(self, client):
        response = client.post(
            "/api/v1/health/analysis",
            json={
                "profile": {"gender": "男"},
                "metrics": {"uric_acid": 300.0, "unknown_metric": "x"},
            },
        )
        assert response.status_code == 200
        assert "存在未识别的指标字段，已被忽略" in response.json()["notices"]

    @pytest.mark.parametrize(
        "payload,expected_message",
        [
            ({}, "请至少提供基本档案或一项体检指标"),
            ({"profile": {}, "metrics": {}}, "请至少提供基本档案或一项体检指标"),
            ({"profile": {"age": 20}, "metrics": {"uric_acid": 450.0}}, "请补充性别信息以完成该项判定"),
            ({"profile": {"gender": "未知"}}, "性别取值不合法"),
            ({"profile": {"gender": "男", "height": 999.0}}, "身高数值异常，请核对"),
            ({"profile": {"gender": "男", "age": 300}}, "年龄数值异常，请核对"),
            ({"profile": {"gender": "男", "weight": 500.0}}, "体重数值异常，请核对"),
            ({"profile": {"gender": "男"}, "metrics": {"h_pylori": "不清楚"}}, "检测结果取值不合法"),
        ],
    )
    def test_validation_errors_return_chinese_messages(self, client, payload, expected_message):
        response = client.post("/api/v1/health/analysis", json=payload)
        assert response.status_code == 422
        messages = [item["msg"] for item in response.json()["detail"]]
        assert expected_message in messages


class TestPerformanceAndConcurrency:
    """7.7 性能与并发验证：单次 ≤ 50ms、50 并发无计算错误。"""

    def test_single_call_latency(self, engine):
        profile = make_profile()
        metrics = make_metrics(**NORMAL_METRICS)
        engine.analyze(profile, metrics)
        durations = []
        for _ in range(20):
            start = time.perf_counter()
            engine.analyze(profile, metrics)
            durations.append(time.perf_counter() - start)
        assert sum(durations) / len(durations) <= 0.05

    def test_50_concurrent_analyses(self, engine):
        profile = make_profile(height=175.0, weight=85.0)
        metrics = make_metrics(uric_acid=450.0)
        with ThreadPoolExecutor(max_workers=50) as executor:
            results = list(executor.map(lambda _: engine.analyze(profile, metrics), range(50)))
        assert len(results) == 50
        for result in results:
            assert tag_codes(result) == ["hyperuricemia", "overweight"]
            assert result.disclaimer == DISCLAIMER_TEXT