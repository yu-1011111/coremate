"""饮食与运动推荐引擎验收测试（tasks.md 任务组 8）。

覆盖：数据接入、标签映射、饮食推荐组合验收（高尿酸+超重午餐）、
运动推荐组合验收（高尿酸+超重操场）、选课建议、API 层集成、
确定性与性能并发。组合验收基准对齐 spec 5.3.1.14 / 5.4.1.12。
"""

import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from app.data.loader import load_records
from app.data.rec_catalog import RecCatalog, _parse_level, create_default_catalog, _dedupe_by_id
from app.main import create_app
from app.rules.diet import create_default_diet_engine
from app.rules.exercise import create_default_exercise_engine
from app.rules.rec_schemas import (
    DISCLAIMER_TEXT,
    INVALID_WEATHER_NOTICE,
    Level,
    MealTime,
    Scene,
    UNKNOWN_TAG_NOTICE,
)
from app.rules.sports_course import create_default_sports_course_advisor
from app.rules.tag_mapping import KNOWN_HEALTH_TAGS, map_health_tags

COMBO_TAGS = ["高尿酸", "超重"]
COMBO_CROWD = frozenset({"高尿酸", "减脂", "超重"})


@pytest.fixture(scope="session")
def catalog():
    return create_default_catalog()


@pytest.fixture(scope="session")
def diet_engine():
    return create_default_diet_engine()


@pytest.fixture(scope="session")
def exercise_engine():
    return create_default_exercise_engine()


@pytest.fixture(scope="session")
def course_advisor():
    return create_default_sports_course_advisor()


@pytest.fixture(scope="session")
def client():
    return TestClient(create_app())


class TestDataIngestion:
    """8.1 数据接入验收：条目数、多值解析、重复 ID 拦截、非法档位归一。"""

    def test_table_row_counts(self, catalog):
        assert len(catalog.actions) == 82
        assert len(catalog.run_courses) == 8
        assert len(catalog.dishes) == 182
        assert len(catalog.drinks) == 41
        assert len(catalog.sport_courses) == 17

    def test_multi_value_field_parsing(self, catalog):
        a020 = next(a for a in catalog.actions if a.action_id == "A020")
        assert a020.block_for == frozenset({"肠胃敏感", "贫血", "超重", "高尿酸", "高血压"})
        assert a020.scene_tags == frozenset({"体测突击", "操场"})
        dish = next(d for d in catalog.dishes if d.dish_id == "D001")
        assert dish.meal_time == frozenset({"lunch", "dinner"})
        assert dish.avoid_for == frozenset({"减脂", "高血压", "高血脂"})

    def test_run_course_id_prefix_split(self, catalog):
        assert all(c.course_id.startswith("R") for c in catalog.run_courses)
        assert all(a.action_id.startswith("A") for a in catalog.actions)
        r_ids = {c.course_id for c in catalog.run_courses}
        assert r_ids == {f"R{i:03d}" for i in range(1, 9)}

    def test_duplicate_id_intercepted(self):
        table_text = "\n".join(
            [
                "| action_id | name_cn | block_for |",
                "| --- | --- | --- |",
                "| A001 | 动作甲 |  |",
                "| A001 | 动作乙 |  |",
            ]
        )
        records = load_records(table_text)
        assert len(records) == 2

        class Item:
            def __init__(self, action_id: str, name: str) -> None:
                self.action_id = action_id
                self.name = name

        items = [Item(r["action_id"], r["name_cn"] or "") for r in records]
        deduped = _dedupe_by_id(items, lambda i: i.action_id, "测试表")
        assert len(deduped) == 1
        assert deduped[0].name == "动作甲"

    def test_invalid_level_normalized(self):
        assert _parse_level("high") is Level.HIGH
        assert _parse_level("mid") is Level.MID
        assert _parse_level("low") is Level.LOW
        assert _parse_level("ultra") is None
        assert _parse_level(None) is None

    def test_load_records_rejects_text_without_header(self):
        with pytest.raises(ValueError):
            load_records("没有表头的文本")

    def test_catalog_process_level_cache(self, catalog):
        assert create_default_catalog() is catalog

    def test_empty_table_degradation_notice(self):
        empty = RecCatalog(dishes=[], drinks=[], actions=[], run_courses=[], sport_courses=[])
        assert empty.dishes == ()
        assert empty.sport_courses == ()


class TestTagMapping:
    """8.2 标签映射验收：双词映射、合并去重、未知标签、空标签。"""

    def test_dual_word_mapping_for_overweight(self):
        mapped = map_health_tags(COMBO_TAGS)
        assert mapped.crowd_tags == COMBO_CROWD

    def test_lipid_tags_merge_to_same_set(self):
        two = map_health_tags(["高胆固醇", "高甘油三酯"])
        one = map_health_tags(["高胆固醇"])
        assert two.crowd_tags == one.crowd_tags == frozenset({"高血脂"})

    def test_unknown_tags_collected(self):
        mapped = map_health_tags(["高尿酸", "夜猫子"])
        assert mapped.crowd_tags == frozenset({"高尿酸"})
        assert mapped.unknown_tags == ("夜猫子",)

    def test_empty_tags_for_healthy_user(self):
        mapped = map_health_tags([])
        assert mapped.crowd_tags == frozenset()
        assert mapped.unknown_tags == ()

    def test_mapping_covers_all_14_health_tags(self):
        from app.rules.tag_mapping import HEALTH_TAG_TO_CROWD_TAGS

        assert KNOWN_HEALTH_TAGS == frozenset(HEALTH_TAG_TO_CROWD_TAGS)
        assert len(HEALTH_TAG_TO_CROWD_TAGS) == 15


class TestDietRecommendation:
    """8.3 饮食推荐组合验收：高尿酸 + 超重 + 午餐（spec 5.3.1.14 基准）。"""

    def test_combo_avoids_high_purine_and_high_oil(self, catalog, diet_engine):
        result = diet_engine.recommend(COMBO_TAGS, MealTime.LUNCH)
        for dish in result.recommended_dishes:
            record = next(d for d in catalog.dishes if d.dish_id == dish.dish_id)
            assert record.purine_level is not Level.HIGH, f"{dish.name} 不应为高嘌呤"
            assert record.oil_level is not Level.HIGH, f"{dish.name} 不应为高油脂"

    def test_combo_avoided_covers_both_hits(self, diet_engine):
        result = diet_engine.recommend(COMBO_TAGS, MealTime.LUNCH)
        avoided_ids = {d.dish_id for d in result.avoided_dishes}
        for dish_id in ("D048", "D062", "D132", "D001", "D061", "D094"):
            assert dish_id in avoided_ids, f"{dish_id} 应进入避雷清单"

    def test_combo_no_intersection_and_reasons(self, diet_engine):
        result = diet_engine.recommend(COMBO_TAGS, MealTime.LUNCH)
        recommended_ids = {d.dish_id for d in result.recommended_dishes}
        avoided_ids = {d.dish_id for d in result.avoided_dishes}
        assert recommended_ids & avoided_ids == set()
        assert all(d.reason for d in result.recommended_dishes)
        assert all(d.reason for d in result.avoided_dishes)

    def test_combo_cola_blocked_with_swap(self, diet_engine):
        result = diet_engine.recommend(COMBO_TAGS, MealTime.LUNCH)
        cola = next(d for d in result.blocked_drinks if d.drink_id == "K011")
        assert cola.name == "可乐"
        assert cola.swap_to == "无糖可乐"
        assert "建议改选" in cola.reason

    def test_combo_disclaimer_and_recommendation_notices(self, diet_engine):
        result = diet_engine.recommend(COMBO_TAGS, MealTime.LUNCH)
        assert result.disclaimer == "本建议仅供日常健康管理参考，不具备临床诊断效力"
        assert result.recommended_dishes, "组合场景推荐清单应为非空"
        assert UNKNOWN_TAG_NOTICE not in result.notices

    def test_honey_water_blocked_with_default_swap(self, diet_engine):
        result = diet_engine.recommend(["高血糖风险"], MealTime.BREAKFAST)
        honey = next(d for d in result.blocked_drinks if d.drink_id == "K041")
        assert honey.name == "蜂蜜水"
        assert honey.swap_to == "白开水"

    def test_reject_dishes_never_recommended(self, diet_engine):
        for meal in (MealTime.BREAKFAST, MealTime.LUNCH, MealTime.SNACK):
            result = diet_engine.recommend(["偏瘦"], meal)
            recommended_ids = {d.dish_id for d in result.recommended_dishes}
            assert "D004" not in recommended_ids
            assert "D112" not in recommended_ids

    def test_safety_priority_sweet_sour_pork(self, diet_engine):
        result = diet_engine.recommend(["高血糖风险", "高尿酸"], MealTime.LUNCH)
        recommended_ids = {d.dish_id for d in result.recommended_dishes}
        avoided_ids = {d.dish_id for d in result.avoided_dishes}
        assert "D072" not in recommended_ids
        assert "D072" in avoided_ids

    def test_healthy_user_gets_universal_recommendation(self, diet_engine):
        result = diet_engine.recommend([], MealTime.LUNCH)
        assert result.avoided_dishes == []
        assert result.blocked_drinks == []
        assert result.caution_notices == []
        assert len(result.recommended_dishes) >= 3
        assert all("禁忌" in d.reason or "常规" in d.reason for d in result.recommended_dishes)

    def test_meal_candidates_scoped(self, catalog, diet_engine):
        result = diet_engine.recommend(["高尿酸"], MealTime.BREAKFAST)
        involved_ids = {d.dish_id for d in result.recommended_dishes} | {
            d.dish_id for d in result.avoided_dishes
        }
        for dish_id in involved_ids:
            record = next(d for d in catalog.dishes if d.dish_id == dish_id)
            assert "breakfast" in record.meal_time

    def test_unknown_tag_notice(self, diet_engine):
        result = diet_engine.recommend(["高尿酸", "夜猫子"], MealTime.LUNCH)
        assert UNKNOWN_TAG_NOTICE in result.notices

    def test_recommend_reason_mentions_crowd_tag(self, diet_engine):
        result = diet_engine.recommend(["高尿酸"], MealTime.LUNCH)
        assert result.recommended_dishes
        assert any("高尿酸" in d.reason for d in result.recommended_dishes)


class TestExerciseRecommendation:
    """8.4 运动推荐组合验收：高尿酸 + 超重（spec 5.4.1.12 基准）。"""

    def test_combo_avoids_high_impact_and_blocked(self, catalog, exercise_engine):
        for scene in (Scene.PLAYGROUND, Scene.TEST_CRASH, Scene.DORM_QUIET):
            result = exercise_engine.recommend(COMBO_TAGS, scene, 30)
            for action in result.recommended_actions:
                record = next(a for a in catalog.actions if a.action_id == action.action_id)
                assert record.impact is not Level.HIGH, f"{action.name} 不应为高冲击"
                assert record.block_for & COMBO_CROWD == frozenset(), f"{action.name} 命中 block_for"

    def test_jump_squat_avoided_with_fallback(self, exercise_engine):
        result = exercise_engine.recommend(COMBO_TAGS, Scene.PLAYGROUND, 30)
        a020 = next(a for a in result.avoided_actions if a.action_id == "A020")
        assert a020.name == "跳跃深蹲"
        assert "冲击" in a020.reason
        suggestion = next(f for f in result.fallback_suggestions if f.original == "跳跃深蹲")
        assert suggestion.fallback == "鞠躬蹲"

    def test_duration_constraint_filters_actions_and_courses(self, exercise_engine):
        result = exercise_engine.recommend(["超重"], Scene.PLAYGROUND, 10)
        assert all(a.est_minutes is not None and a.est_minutes <= 10 for a in result.recommended_actions)
        assert all(c.duration_min is not None and c.duration_min <= 10 for c in result.run_courses)
        assert "R001" not in {c.course_id for c in result.run_courses}

    def test_duration_too_short_notice(self, exercise_engine):
        result = exercise_engine.recommend([], Scene.PLAYGROUND, 1)
        assert result.recommended_actions == []
        assert "可用时长过短" in "".join(result.notices)

    def test_rainy_weather_blocks_run_courses_with_swap(self, exercise_engine):
        result = exercise_engine.recommend(["超重"], Scene.PLAYGROUND, 30, "雨")
        r001 = next(c for c in result.run_courses if c.course_id == "R001")
        assert r001.weather_blocked is True
        assert r001.weather_swap == "原地踏步 20 分钟 + 靠墙高抬腿"

    def test_rainy_weather_r006_special_swap(self, exercise_engine):
        result = exercise_engine.recommend([], Scene.TEST_CRASH, 30, "雨")
        r006 = next(c for c in result.run_courses if c.course_id == "R006")
        assert r006.weather_blocked is True
        assert r006.weather_swap == "延后至天气合适日，不做室内替代"

    def test_sunny_weather_does_not_block(self, exercise_engine):
        result = exercise_engine.recommend(["超重"], Scene.PLAYGROUND, 30, "晴")
        assert all(c.weather_blocked is False for c in result.run_courses)
        assert all(c.weather_swap is None for c in result.run_courses)

    def test_missing_weather_does_not_block(self, exercise_engine):
        result = exercise_engine.recommend(["超重"], Scene.PLAYGROUND, 30)
        assert all(c.weather_blocked is False for c in result.run_courses)

    def test_high_temperature_prefix_match(self, exercise_engine):
        result = exercise_engine.recommend([], Scene.PLAYGROUND, 60, "高温")
        r008 = next(c for c in result.run_courses if c.course_id == "R008")
        assert r008.weather_blocked is True

    def test_overweight_playground_course_advice(self, exercise_engine):
        result = exercise_engine.recommend(["超重"], Scene.PLAYGROUND, 30)
        avoided_ids = {a.action_id for a in result.avoided_actions}
        assert {f"R{i:03d}" for i in range(2, 9)} <= avoided_ids
        r001 = next(c for c in result.run_courses if c.course_id == "R001")
        assert "跑走交替" in r001.name
        caution = next(c for c in result.caution_notices if c.target == r001.name)
        assert "跑走交替控制冲击累计量" in caution.reason

    def test_hypertension_action_blocked(self, exercise_engine):
        result = exercise_engine.recommend(["高血压"], Scene.PLAYGROUND, 30)
        a007 = next(a for a in result.avoided_actions if a.action_id == "A007")
        assert a007.name == "尺蠖爬行"
        assert "颅内压" in a007.reason

    def test_hyperuricemia_dorm_recommendation(self, exercise_engine):
        result = exercise_engine.recommend(["高尿酸"], Scene.DORM_QUIET, 20)
        a023 = next(a for a in result.recommended_actions if a.action_id == "A023")
        assert a023.name == "侧卧股四头肌拉伸"
        assert a023.est_minutes == 2
        assert "关节活动度" in a023.reason

    def test_invalid_weather_notice(self, exercise_engine):
        result = exercise_engine.recommend(["超重"], Scene.PLAYGROUND, 30, "台风")
        assert INVALID_WEATHER_NOTICE in result.notices
        assert all(c.weather_blocked is False for c in result.run_courses)

    def test_healthy_user_gets_universal_recommendation(self, exercise_engine):
        result = exercise_engine.recommend([], Scene.DORM_QUIET, 20)
        assert result.avoided_actions == []
        assert result.recommended_actions

    def test_disclaimer_always_present(self, exercise_engine):
        result = exercise_engine.recommend(COMBO_TAGS, Scene.PLAYGROUND, 30)
        assert result.disclaimer == DISCLAIMER_TEXT


class TestSportsCourseRecommendation:
    """8.5 选课建议验收：超重场景三路判定。"""

    def test_overweight_course_advice(self, course_advisor):
        result = course_advisor.recommend(["超重"])
        recommended_ids = {c.course_id for c in result.recommended_courses}
        avoided_ids = {c.course_id for c in result.avoided_courses}
        assert "S02" in recommended_ids
        assert "S03" in avoided_ids
        caution_targets = {c.target for c in result.caution_notices}
        assert "羽毛球" in caution_targets
        assert recommended_ids & avoided_ids == set()

    def test_overweight_pingpong_reason(self, course_advisor):
        result = course_advisor.recommend(["超重"])
        s02 = next(c for c in result.recommended_courses if c.course_id == "S02")
        assert "无跳跃冲击" in s02.reason

    def test_overweight_basketball_reason(self, course_advisor):
        result = course_advisor.recommend(["超重"])
        s03 = next(c for c in result.avoided_courses if c.course_id == "S03")
        assert "跳跃" in s03.reason or "冲击" in s03.reason

    def test_healthy_user_all_courses(self, course_advisor):
        result = course_advisor.recommend([])
        assert len(result.recommended_courses) == 17
        assert result.avoided_courses == []
        assert result.caution_notices == []

    def test_disclaimer_always_present(self, course_advisor):
        result = course_advisor.recommend(["贫血风险"])
        assert result.disclaimer == DISCLAIMER_TEXT


class TestRecommendApi:
    """8.6 API 层集成：三个端点 + 422 中文提示。"""

    def test_diet_endpoint_returns_full_result(self, client):
        response = client.post(
            "/api/v1/recommend/diet",
            json={"tags": COMBO_TAGS, "meal": "lunch"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["disclaimer"] == DISCLAIMER_TEXT
        assert body["recommended_dishes"]
        assert body["avoided_dishes"]
        blocked_names = {d["name"] for d in body["blocked_drinks"]}
        assert "可乐" in blocked_names

    def test_exercise_endpoint_returns_full_result(self, client):
        response = client.post(
            "/api/v1/recommend/exercise",
            json={"tags": COMBO_TAGS, "scene": "操场", "available_minutes": 30, "weather": "雨"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["disclaimer"] == DISCLAIMER_TEXT
        fallbacks = {f["original"] for f in body["fallback_suggestions"]}
        assert "跳跃深蹲" in fallbacks
        r001 = next(c for c in body["run_courses"] if c["course_id"] == "R001")
        assert r001["weather_blocked"] is True

    def test_sports_course_endpoint_returns_full_result(self, client):
        response = client.post("/api/v1/recommend/sports-course", json={"tags": ["超重"]})
        assert response.status_code == 200
        body = response.json()
        recommended_ids = {c["course_id"] for c in body["recommended_courses"]}
        assert "S02" in recommended_ids

    def test_health_check_untouched(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "service": "coremate-medical-rules-engine"}

    @pytest.mark.parametrize(
        "payload,expected_message",
        [
            ({"tags": []}, "请提供用餐餐次"),
            ({"tags": [], "meal": "brunch"}, "餐次取值不合法"),
            ({"tags": [], "available_minutes": 30}, "请提供用餐餐次"),
            ({"tags": [], "scene": "操场", "available_minutes": 30}, "请提供用餐餐次"),
        ],
    )
    def test_diet_validation_errors(self, client, payload, expected_message):
        response = client.post("/api/v1/recommend/diet", json=payload)
        assert response.status_code == 422
        messages = [item["msg"] for item in response.json()["detail"]]
        assert expected_message in messages

    @pytest.mark.parametrize(
        "payload,expected_message",
        [
            ({"tags": [], "available_minutes": 30}, "请提供运动场景"),
            ({"tags": [], "scene": "健身房", "available_minutes": 30}, "场景取值不合法"),
            ({"tags": [], "scene": "操场"}, "请提供可用运动时长"),
            ({"tags": [], "scene": "操场", "available_minutes": 0}, "可用运动时长数值异常"),
            ({"tags": [], "scene": "操场", "available_minutes": 999}, "可用运动时长数值异常"),
        ],
    )
    def test_exercise_validation_errors(self, client, payload, expected_message):
        response = client.post("/api/v1/recommend/exercise", json=payload)
        assert response.status_code == 422
        messages = [item["msg"] for item in response.json()["detail"]]
        assert expected_message in messages

    def test_exercise_weather_is_loose_not_422(self, client):
        response = client.post(
            "/api/v1/recommend/exercise",
            json={"tags": ["超重"], "scene": "操场", "available_minutes": 30, "weather": "台风"},
        )
        assert response.status_code == 200
        assert "天气取值不合法，已忽略" in response.json()["notices"]

    def test_unknown_tag_notice_via_api(self, client):
        response = client.post(
            "/api/v1/recommend/diet",
            json={"tags": ["夜猫子"], "meal": "lunch"},
        )
        assert response.status_code == 200
        assert "存在未识别的健康标签，已被忽略" in response.json()["notices"]


class TestDeterminismAndPerformance:
    """8.7 确定性、性能与并发验证。"""

    def test_diet_same_input_same_output(self, diet_engine):
        dumps = [diet_engine.recommend(COMBO_TAGS, MealTime.LUNCH).model_dump() for _ in range(3)]
        assert dumps[0] == dumps[1] == dumps[2]

    def test_exercise_same_input_same_output(self, exercise_engine):
        dumps = [
            exercise_engine.recommend(COMBO_TAGS, Scene.PLAYGROUND, 30, "雨").model_dump()
            for _ in range(3)
        ]
        assert dumps[0] == dumps[1] == dumps[2]

    def test_single_call_latency(self, diet_engine, exercise_engine):
        diet_engine.recommend(COMBO_TAGS, MealTime.LUNCH)
        exercise_engine.recommend(COMBO_TAGS, Scene.PLAYGROUND, 30)
        durations = []
        for _ in range(20):
            start = time.perf_counter()
            diet_engine.recommend(COMBO_TAGS, MealTime.LUNCH)
            exercise_engine.recommend(COMBO_TAGS, Scene.PLAYGROUND, 30)
            durations.append(time.perf_counter() - start)
        assert sum(durations) / len(durations) <= 0.05

    def test_50_concurrent_recommendations(self, diet_engine, exercise_engine):
        with ThreadPoolExecutor(max_workers=50) as executor:
            diet_futures = [
                executor.submit(diet_engine.recommend, COMBO_TAGS, MealTime.LUNCH) for _ in range(50)
            ]
            exercise_futures = [
                executor.submit(exercise_engine.recommend, COMBO_TAGS, Scene.PLAYGROUND, 30)
                for _ in range(50)
            ]
            diet_results = [f.result() for f in diet_futures]
            exercise_results = [f.result() for f in exercise_futures]
        assert len(diet_results) == 50
        assert len(exercise_results) == 50
        for result in diet_results:
            assert result.disclaimer == DISCLAIMER_TEXT
            assert result.recommended_dishes
        for result in exercise_results:
            assert result.disclaimer == DISCLAIMER_TEXT

class TestRecommendedActionCueField:
    """exercise_cue_display · 后端契约锚点（独立成类）。"""

    def test_cue_field_defaults_none(self):
        from app.rules.rec_schemas import RecommendedAction

        action = RecommendedAction(action_id="A007", name="尺蠖爬行", reason="r")
        assert action.cue is None

    def test_existing_fields_unchanged_with_cue_default(self):
        from app.rules.rec_schemas import RecommendedAction

        action = RecommendedAction(
            action_id="A007", name="尺蠖爬行", reason="r", est_minutes=5
        )
        dumped = action.model_dump()
        assert set(dumped) == {"action_id", "name", "reason", "est_minutes", "cue"}
        assert dumped["action_id"] == "A007"
        assert dumped["name"] == "尺蠖爬行"
        assert dumped["reason"] == "r"
        assert dumped["est_minutes"] == 5
        assert dumped["cue"] is None

    def test_cue_value_kept_verbatim(self):
        from app.rules.rec_schemas import RecommendedAction

        action = RecommendedAction(
            action_id="A007",
            name="尺蠖爬行",
            reason="r",
            cue="站立前屈手撑地，手前爬至平板再双脚前移",
        )
        assert action.cue == "站立前屈手撑地，手前爬至平板再双脚前移"