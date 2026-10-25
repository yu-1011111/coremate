"""心理干预模块验收测试（tasks.md 任务组 9）。

覆盖：数据接入、便签自查引擎、情绪信号识别引擎、资源推荐、
M4 危机链路、抑制机制、API 层集成与 DFX。
用户三个验收锚点分别落位：危机组（"我想死"→M4 无饮食运动建议）、
自查组（SC3+3 个月以上→专业入口）、资源组（M1→K25-K28+音频）。
"""

import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from app.data.loader import load_records
from app.data.mood_loader import AUDIO_DIR, create_default_mood_catalog
from app.main import create_app
from app.rules.crisis import (
    CRISIS_CLOSING,
    CRISIS_OPENING,
    GUIDE_TEXT,
    create_default_crisis_responder,
)
from app.rules.diet import create_default_diet_engine
from app.rules.exercise import create_default_exercise_engine
from app.rules.mood_schemas import (
    DISCLAIMER_TEXT,

    BehaviorEvent,
    DurationLevel,
    RefType,
    SelfcheckInvalidInputError,
    SignalEmptyInputError,
)
from app.rules.rec_schemas import SUPPRESSED_NOTICE
from app.rules.selfcheck import create_default_selfcheck_engine
from app.rules.signal_engine import create_default_signal_engine

COMBO = ["高尿酸", "超重"]


@pytest.fixture(scope="session")
def catalog():
    return create_default_mood_catalog()


@pytest.fixture(scope="session")
def selfcheck_engine():
    return create_default_selfcheck_engine()


@pytest.fixture(scope="session")
def signal_engine():
    return create_default_signal_engine()


@pytest.fixture(scope="session")
def crisis_responder():
    return create_default_crisis_responder()


@pytest.fixture(scope="session")
def client():
    return TestClient(create_app())


class TestDataIngestion:
    """9.1 数据接入验收：条目数、多值解析、ref 识别、电话保真、缓存。"""

    def test_table_row_counts(self, catalog):
        assert len(catalog.kb_entries) == 40
        assert len(catalog.selfcheck_items) == 9
        assert len(catalog.signals) == 14
        assert len(catalog.mood_resources) == 31
        assert len(catalog.campus_resources) == 7

    def test_multi_value_field_parsing(self, catalog):
        s01 = catalog.get_signal("S01")
        assert s01.keywords == frozenset({"不想活了", "想死", "自杀", "割腕", "跳楼", "自残"})
        assert s01.weight == 99
        assert s01.target_mood == "M4"
        sc3 = catalog.get_selfcheck("SC3")
        assert sc3.kb_ids == frozenset({"K21", "K23", "K24"})
        assert sc3.referral_ids == frozenset({"C04", "C06"})
        assert sc3.has_referral is True
        k37 = catalog.get_kb("K37")
        assert k37.mood == frozenset({"M1", "SLEEP"})
        k15 = catalog.get_kb("K15")
        assert len(k15.links) == 2

    def test_signal_derived_fields(self, catalog):
        assert catalog.get_signal("S02").min_consecutive_days == 3
        assert catalog.get_signal("S03").min_consecutive_days == 3
        assert catalog.get_signal("S13").min_consecutive_days == 3
        assert catalog.get_signal("S09").sc_ids == frozenset({"SC1", "SC4"})
        assert catalog.get_signal("S10").sc_ids == frozenset({"SC3"})
        assert catalog.get_signal("S11").sc_ids == frozenset({"SC7"})
        assert catalog.get_signal("S12").sc_ids == frozenset({"SC9"})

    @pytest.mark.parametrize(
        "res_id,ref_type,object_id",
        [
            ("MR01", RefType.ACTION, "A079"),
            ("MR04", RefType.AUDIO, "篝火.mp3"),
            ("MR06", RefType.FOOD, "L20"),
            ("MR08", RefType.DRINK, "K008"),
            ("MR15", RefType.DRINK, "K001"),
            ("MR25", RefType.CAMPUS, "C01"),
            ("MR30", RefType.SCRIPT, None),
        ],
    )
    def test_ref_cross_library_resolution(self, catalog, res_id, ref_type, object_id):
        record = next(m for m in catalog.mood_resources if m.res_id == res_id)
        assert record.ref is not None
        assert record.ref.ref_type is ref_type
        assert record.ref.object_id == object_id

    def test_k_prefix_disambiguation(self, catalog):
        mr08 = next(m for m in catalog.mood_resources if m.res_id == "MR08")
        assert mr08.ref.ref_type is RefType.DRINK and mr08.ref.object_id == "K008"
        mr15 = next(m for m in catalog.mood_resources if m.res_id == "MR15")
        assert mr15.ref.ref_type is RefType.DRINK and mr15.ref.object_id == "K001"
        k22 = catalog.get_kb("K22")
        ref_types = {(l.ref_type, l.object_id) for l in k22.links}
        assert (RefType.RUN_COURSE, "R002") in ref_types
        assert (RefType.MOOD_KB, "K18") in ref_types

    def test_k40_link_audio_and_run_course(self, catalog):
        k40 = catalog.get_kb("K40")
        refs = [(l.ref_type, l.object_id) for l in k40.links]
        assert (RefType.AUDIO, "篝火.mp3") in refs
        assert (RefType.AUDIO, "晚风.mp3") in refs
        assert (RefType.RUN_COURSE, "R001") in refs

    def test_phone_fidelity(self, catalog):
        c01 = catalog.get_campus("C01")
        assert c01.phone == "0516-83105638"
        assert c01.hours == "24 小时"
        c04 = catalog.get_campus("C04")
        assert c04.phone == "0516-83689615 / 0516-83689877"
        assert c04.location == "大学生活动中心 108 室、111 室"
        c06 = catalog.get_campus("C06")
        assert c06.phone is None
        assert c06.location == "应用商店搜索「苏心」"

    def test_duplicate_id_intercepted(self):
        table_text = "\n".join(
            [
                "| kb_id | method |",
                "| --- | --- |",
                "| K01 | 甲法 |",
                "| K01 | 乙法 |",
            ]
        )
        records = load_records(table_text)
        assert len(records) == 2

    def test_audio_unavailable_when_file_missing(self, catalog, tmp_path, monkeypatch, caplog):
        import logging

        import app.data.mood_loader as mood_loader

        monkeypatch.setattr(mood_loader, "AUDIO_DIR", tmp_path)
        monkeypatch.setattr(mood_loader, "_DEFAULT_CATALOG", None)
        with caplog.at_level(logging.WARNING, logger="coremate.data"):
            rebuilt = mood_loader.create_default_mood_catalog()
        mr04 = next(m for m in rebuilt.mood_resources if m.res_id == "MR04")
        assert mr04.ref.ref_type is RefType.AUDIO
        assert mr04.ref.available is False
        assert "音频文件缺失，引用降级" in caplog.text
        assert "篝火.mp3" in caplog.text
        assert mr04.name == "篝火声"
        assert mr04.mood_level == "M1"

    def test_process_level_cache(self, catalog):
        assert create_default_mood_catalog() is catalog

    def test_audio_all_references_available(self, catalog):
        audio_refs = [
            record.ref
            for record in catalog.mood_resources
            if record.ref is not None and record.ref.ref_type is RefType.AUDIO
        ]
        audio_refs.extend(
            link for link in catalog.get_kb("K40").links if link.ref_type is RefType.AUDIO
        )
        assert len(audio_refs) == 10
        for ref in audio_refs:
            assert ref.available is True, f"{ref.object_id} 应可用"
            assert (AUDIO_DIR / ref.object_id).is_file(), f"{ref.object_id} 文件应存在"

    def test_audio_dir_resolves_to_project_root(self):
        assert (AUDIO_DIR / "README.txt").is_file()


class TestSelfcheckEngine:
    """9.2 便签自查引擎验收（用户测试 2 锚点）。"""

    def test_sc3_over_3m_reaches_professional_entry(self, selfcheck_engine):
        result = selfcheck_engine.evaluate(["SC3"], DurationLevel.OVER_3M)
        assert result.mood_level == "M3"
        assert {k.kb_id for k in result.kb_recommendations} == {"K21", "K23", "K24"}
        referral_ids = {r.res_id for r in result.referrals}
        assert referral_ids == {"C04", "C06"}
        c04 = next(r for r in result.referrals if r.res_id == "C04")
        assert c04.contact == "0516-83689615 / 0516-83689877"
        assert "困扰持续较久" in "".join(result.notices)

    def test_base_mood_takes_highest(self, selfcheck_engine):
        result = selfcheck_engine.evaluate(["SC1", "SC5"], DurationLevel.UNDER_2W)
        assert result.mood_level == "M3"

    def test_sc1_under_2w_m1_with_k25_to_k28(self, selfcheck_engine):
        result = selfcheck_engine.evaluate(["SC1"], DurationLevel.UNDER_2W)
        assert result.mood_level == "M1"
        methods = {k.kb_id: k.method for k in result.kb_recommendations}
        assert methods["K25"] == "没事找事法"
        assert methods["K27"] == "记名字法"
        assert methods["K26"] == "间接接触法"
        assert methods["K28"] == "自报家门法"
        assert result.referrals == []

    def test_sc1_over_3m_fallback_referral(self, selfcheck_engine):
        result = selfcheck_engine.evaluate(["SC1"], DurationLevel.OVER_3M)
        assert result.mood_level == "M2"
        assert {r.res_id for r in result.referrals} == {"C04"}
        assert "困扰持续较久" in "".join(result.notices)

    def test_sc7_links_a079(self, selfcheck_engine):
        result = selfcheck_engine.evaluate(["SC7"], DurationLevel.UNDER_2W)
        assert {k.kb_id for k in result.kb_recommendations} == {"K17", "K18", "K19"}
        k18 = next(k for k in result.kb_recommendations if k.kb_id == "K18")
        assert any(
            link.ref_type is RefType.ACTION and link.object_id == "A079" for link in k18.links
        )
        assert {r.res_id for r in result.referrals} == {"C04"}

    def test_all_checked_capped_at_m3(self, selfcheck_engine):
        result = selfcheck_engine.evaluate(
            [f"SC{i}" for i in range(1, 10)], DurationLevel.OVER_3M
        )
        assert result.mood_level == "M3"

    def test_unknown_sc_ignored_with_notice(self, selfcheck_engine):
        result = selfcheck_engine.evaluate(["SC1", "SC99"], DurationLevel.UNDER_2W)
        assert result.mood_level == "M1"
        assert "存在未识别的自查条目，已被忽略" in result.notices

    def test_all_unknown_raises_invalid_input(self, selfcheck_engine):
        with pytest.raises(SelfcheckInvalidInputError):
            selfcheck_engine.evaluate(["SC99"], DurationLevel.UNDER_2W)

    def test_duration_shift_m1_to_3m(self, selfcheck_engine):
        result = selfcheck_engine.evaluate(["SC3"], DurationLevel.M1_TO_3M)
        assert result.mood_level == "M3"

    def test_disclaimer_consistent(self, selfcheck_engine):
        result = selfcheck_engine.evaluate(["SC1"], DurationLevel.UNDER_2W)
        assert result.disclaimer == DISCLAIMER_TEXT


class TestSignalEngine:
    """9.3 情绪信号识别引擎验收（用户测试 1 引擎部分）。"""

    def test_s01_hard_triggers_m4(self, signal_engine):
        result = signal_engine.analyze("我想死")
        assert result.mood_level == "M4"
        s01 = next(m for m in result.matched_signals if m.signal_id == "S01")
        assert "想死" in s01.hit_keywords

    def test_s01_coexists_with_other_signals(self, signal_engine):
        result = signal_engine.analyze("我想死，最近还总失眠")
        ids = {m.signal_id for m in result.matched_signals}
        assert ids == {"S01", "S06"}
        assert result.mood_level == "M4"

    @pytest.mark.parametrize(
        "text,expected_signals",
        [
            ("最近好累，活着没劲", {"S04"}),
            ("我睡不着，有点焦虑", {"S06"}),
            ("今天上了一节有趣的课", set()),
        ],
    )
    def test_text_keyword_matching(self, signal_engine, text, expected_signals):
        result = signal_engine.analyze(text)
        assert {m.signal_id for m in result.matched_signals} == expected_signals

    def test_weight_accumulation_to_m2(self, signal_engine):
        result = signal_engine.analyze(
            "好累", [BehaviorEvent(event_type="低步数", consecutive_days=3)]
        )
        assert result.total_weight == 6
        assert result.mood_level == "M2"

    def test_target_mood_floor_s04(self, signal_engine):
        result = signal_engine.analyze("好累")
        assert result.total_weight == 4
        assert result.mood_level == "M2"

    def test_self_report_sc3_translates_to_s10(self, signal_engine):
        result = signal_engine.analyze("", [], ["SC3"])
        assert {m.signal_id for m in result.matched_signals} == {"S10"}
        assert result.total_weight == 5
        assert result.mood_level == "M2"

    def test_behavior_events_threshold(self, signal_engine):
        hit = signal_engine.analyze("", [BehaviorEvent(event_type="深夜活跃", consecutive_days=3)])
        assert {m.signal_id for m in hit.matched_signals} == {"S02"}
        miss = signal_engine.analyze("", [BehaviorEvent(event_type="低步数", consecutive_days=2)])
        assert miss.matched_signals == []
        assert miss.mood_level == "无情绪异常"

    def test_no_match_returns_no_issue(self, signal_engine):
        result = signal_engine.analyze("今天上了一节有趣的课")
        assert result.mood_level == "无情绪异常"
        assert result.total_weight == 0
        assert result.mood_resources == []

    def test_empty_input_raises(self, signal_engine):
        with pytest.raises(SignalEmptyInputError):
            signal_engine.analyze("")

    def test_m4_no_resources_attached(self, signal_engine):
        result = signal_engine.analyze("我想死")
        assert result.mood_resources == []

    def test_high_accumulation_capped_at_m3(self, signal_engine):
        result = signal_engine.analyze(
            "不想上学好累没希望崩溃，睡不着失眠紧张害怕焦虑心慌胸闷，烦死了气死了想打架，难受孤单想哭"
        )
        assert result.total_weight >= 9
        assert result.mood_level == "M3"


class TestMoodResources:
    """9.4 资源推荐验收（用户测试 3 锚点）。"""

    def test_m1_resources_full_eight_sorted(self, selfcheck_engine):
        result = selfcheck_engine.evaluate(["SC1"], DurationLevel.UNDER_2W)
        assert [r.res_id for r in result.mood_resources] == [
            "MR01", "MR04", "MR06", "MR02", "MR05", "MR07", "MR03", "MR08",
        ]
        priorities = [r.priority for r in result.mood_resources]
        assert priorities == sorted(priorities)

    def test_m1_resources_refs_attached(self, selfcheck_engine):
        result = selfcheck_engine.evaluate(["SC1"], DurationLevel.UNDER_2W)
        mr01 = next(r for r in result.mood_resources if r.res_id == "MR01")
        assert mr01.ref.ref_type is RefType.ACTION and mr01.ref.object_id == "A079"
        mr04 = next(r for r in result.mood_resources if r.res_id == "MR04")
        assert mr04.ref.ref_type is RefType.AUDIO and mr04.ref.available is True
        mr06 = next(r for r in result.mood_resources if r.res_id == "MR06")
        assert mr06.ref.ref_type is RefType.FOOD and mr06.ref.object_id == "L20"

    def test_m2_resources_exclude_sleep(self, selfcheck_engine):
        result = selfcheck_engine.evaluate(["SC3"], DurationLevel.UNDER_2W)
        assert result.mood_level == "M2"
        ids = {r.res_id for r in result.mood_resources}
        assert ids == {f"MR{i:02d}" for i in range(9, 16)}
        assert not ids & {f"MR{i:02d}" for i in range(21, 25)}

    def test_m3_resources_five_items(self, selfcheck_engine):
        result = selfcheck_engine.evaluate(["SC3"], DurationLevel.OVER_3M)
        assert result.mood_level == "M3"
        ids = {r.res_id for r in result.mood_resources}
        assert ids == {f"MR{i:02d}" for i in range(16, 21)}

    def test_signal_engine_m2_resources(self, signal_engine):
        result = signal_engine.analyze("我睡不着，有点焦虑")
        assert result.mood_level == "M2"
        ids = {r.res_id for r in result.mood_resources}
        assert ids == {f"MR{i:02d}" for i in range(9, 16)}


class TestCrisisResponse:
    """9.5 M4 危机链路验收（用户测试 1 锚点，spec 5.5 全规则）。"""

    def test_crisis_response_full_structure(self, crisis_responder, signal_engine):
        result = signal_engine.analyze("我想死")
        crisis = crisis_responder.respond(result.matched_signals)
        assert crisis.opening == CRISIS_OPENING == "你现在一个人吗？安全吗？"
        assert crisis.closing == CRISIS_CLOSING == "你不用一个人扛。刚才说的这些，听起来已经很难受了。"
        assert crisis.suppression is True
        assert crisis.disclaimer == DISCLAIMER_TEXT

        assert [r.res_id for r in crisis.referrals] == ["C01", "C02", "C03", "C04", "C06", None]
        c01, c02, c03, c04, c06, guide = crisis.referrals
        assert c01.contact == "0516-83105638" and c01.hours == "24 小时"
        assert c02.contact == "12356"
        assert c03.contact == "025-58255200"
        assert c04.contact == "0516-83689615 / 0516-83689877"
        assert "应用商店搜索「苏心」" in (c06.contact or "")
        assert guide.name == "可以联系身边的人"
        assert guide.contact == GUIDE_TEXT

    def test_crisis_suppresses_recommendations(self, crisis_responder, signal_engine):
        result = signal_engine.analyze("我想死")
        crisis = crisis_responder.respond(result.matched_signals)
        assert crisis.diet_recommendations.recommended_dishes == []
        assert crisis.diet_recommendations.avoided_dishes == []
        assert crisis.diet_recommendations.blocked_drinks == []
        assert SUPPRESSED_NOTICE in crisis.diet_recommendations.notices
        assert crisis.exercise_recommendations.recommended_actions == []
        assert crisis.exercise_recommendations.run_courses == []
        assert SUPPRESSED_NOTICE in crisis.exercise_recommendations.notices

    def test_crisis_field_order_disclaimer_last(self, crisis_responder, signal_engine):
        result = signal_engine.analyze("我想死")
        crisis = crisis_responder.respond(result.matched_signals)
        keys = list(crisis.model_dump().keys())
        assert keys.index("opening") < keys.index("referrals") < keys.index("closing")
        assert keys.index("closing") < keys.index("disclaimer")

    def test_crisis_repeat_calls_not_deduplicated(self, crisis_responder, signal_engine):
        result = signal_engine.analyze("我想死")
        first = crisis_responder.respond(result.matched_signals)
        second = crisis_responder.respond(result.matched_signals)
        assert first.model_dump() == second.model_dump()
        assert len(first.referrals) == len(second.referrals) == 6


class TestSuppression:
    """9.6 抑制机制验收：引擎构造 + API 分流。"""

    def test_diet_recommend_suppressed(self):
        result = create_default_diet_engine().recommend_suppressed()
        assert result.recommended_dishes == []
        assert result.avoided_dishes == []
        assert result.recommended_drinks == []
        assert result.blocked_drinks == []
        assert SUPPRESSED_NOTICE in result.notices
        assert result.disclaimer == DISCLAIMER_TEXT

    def test_exercise_recommend_suppressed(self):
        result = create_default_exercise_engine().recommend_suppressed()
        assert result.recommended_actions == []
        assert result.avoided_actions == []
        assert result.fallback_suggestions == []
        assert result.run_courses == []
        assert SUPPRESSED_NOTICE in result.notices
        assert result.disclaimer == DISCLAIMER_TEXT

    def test_api_diet_suppression_flag(self, client):
        response = client.post(
            "/api/v1/recommend/diet",
            json={"tags": COMBO, "meal": "lunch", "suppression": True},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["recommended_dishes"] == []
        assert body["avoided_dishes"] == []
        assert SUPPRESSED_NOTICE in body["notices"]

    def test_api_exercise_suppression_flag(self, client):
        response = client.post(
            "/api/v1/recommend/exercise",
            json={
                "tags": COMBO,
                "scene": "操场",
                "available_minutes": 30,
                "suppression": True,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["recommended_actions"] == []
        assert body["run_courses"] == []
        assert SUPPRESSED_NOTICE in body["notices"]

    def test_api_default_behavior_unchanged(self, client):
        response = client.post(
            "/api/v1/recommend/diet",
            json={"tags": COMBO, "meal": "lunch"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["recommended_dishes"]
        assert body["avoided_dishes"]


class TestMoodApi:
    """9.7 API 层集成与 DFX 验收。"""

    def test_selfcheck_endpoint(self, client):
        response = client.post(
            "/api/v1/mood/selfcheck",
            json={"checked_sc_ids": ["SC3"], "duration": "3 个月以上"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["mood_level"] == "M3"
        assert {r["res_id"] for r in body["referrals"]} == {"C04", "C06"}
        assert body["disclaimer"] == DISCLAIMER_TEXT

    def test_signal_endpoint_m4_with_crisis(self, client):
        response = client.post("/api/v1/mood/signal", json={"text": "我想死"})
        assert response.status_code == 200
        body = response.json()
        assert body["mood_level"] == "M4"
        assert {m["signal_id"] for m in body["matched_signals"]} == {"S01"}
        crisis = body["crisis"]
        assert crisis["opening"] == CRISIS_OPENING
        assert [r["res_id"] for r in crisis["referrals"]] == ["C01", "C02", "C03", "C04", "C06", None]
        assert crisis["suppression"] is True
        assert crisis["diet_recommendations"]["recommended_dishes"] == []
        assert crisis["exercise_recommendations"]["recommended_actions"] == []
        assert body["mood_resources"] == []
        serialized = response.text
        assert "腹式呼吸" not in serialized
        assert "热的蔬菜汤" not in serialized

    def test_signal_endpoint_m2_normal(self, client):
        response = client.post("/api/v1/mood/signal", json={"text": "我睡不着，有点焦虑"})
        assert response.status_code == 200
        body = response.json()
        assert body["mood_level"] == "M2"
        assert body["crisis"] is None
        assert {r["res_id"] for r in body["mood_resources"]} == {f"MR{i:02d}" for i in range(9, 16)}

    @pytest.mark.parametrize(
        "payload,status,message",
        [
            ({"checked_sc_ids": [], "duration": "<2 周"}, 422, "请至少勾选一条自查条目"),
            ({"checked_sc_ids": ["SC1"], "duration": "半年"}, 422, "持续时长取值不合法"),
            ({"checked_sc_ids": ["SC99"], "duration": "<2 周"}, 400, "未识别任何有效自查条目"),
            ({"text": "", "behavior_events": []}, 422, "请提供对话文本或行为事件"),
        ],
    )
    def test_validation_errors_chinese_messages(self, client, payload, status, message):
        endpoint = "/api/v1/mood/selfcheck" if "checked_sc_ids" in payload else "/api/v1/mood/signal"
        response = client.post(endpoint, json=payload)
        assert response.status_code == status
        detail = response.json()["detail"]
        if isinstance(detail, list):
            assert message in " ".join(item.get("msg", "") for item in detail)
        else:
            assert message in detail

    def test_behavior_event_via_api(self, client):
        response = client.post(
            "/api/v1/mood/signal",
            json={"text": "", "behavior_events": [{"event_type": "深夜活跃", "consecutive_days": 3}]},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["mood_level"] == "M1"
        assert {m["signal_id"] for m in body["matched_signals"]} == {"S02"}


class TestDeterminismAndPerformance:
    """9.8 确定性、性能与并发。"""

    def test_selfcheck_same_input_same_output(self, selfcheck_engine):
        dumps = [
            selfcheck_engine.evaluate(["SC3"], DurationLevel.OVER_3M).model_dump()
            for _ in range(3)
        ]
        assert dumps[0] == dumps[1] == dumps[2]

    def test_signal_same_input_same_output(self, signal_engine):
        dumps = [signal_engine.analyze("我想死").model_dump() for _ in range(3)]
        assert dumps[0] == dumps[1] == dumps[2]

    def test_single_call_latency(self, selfcheck_engine, signal_engine, crisis_responder):
        selfcheck_engine.evaluate(["SC3"], DurationLevel.OVER_3M)
        signal_engine.analyze("我睡不着，有点焦虑")
        signal_engine.analyze("我想死")
        durations = []
        for _ in range(20):
            start = time.perf_counter()
            selfcheck_engine.evaluate(["SC3"], DurationLevel.OVER_3M)
            signal_engine.analyze("我睡不着，有点焦虑")
            durations.append(time.perf_counter() - start)
        assert sum(durations) / len(durations) <= 0.05

    def test_50_concurrent_mood_requests(self, signal_engine, crisis_responder):
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [
                executor.submit(signal_engine.analyze, text)
                for text in (["我想死"] * 25 + ["我睡不着，有点焦虑"] * 25)
            ]
            results = [f.result() for f in futures]
        assert len(results) == 50
        m4_count = sum(1 for r in results if r.mood_level == "M4")
        m2_count = sum(1 for r in results if r.mood_level == "M2")
        assert m4_count == 25 and m2_count == 25

    def test_regression_old_suites_untouched(self, client):
        response = client.post(
            "/api/v1/health/analysis",
            json={
                "profile": {"gender": "男", "height": 175.0, "weight": 85.0},
                "metrics": {"uric_acid": 450.0},
            },
        )
        assert response.status_code == 200
        assert [t["name"] for t in response.json()["tags"]] == ["高尿酸", "超重"]

class TestMoodHiddenWords:
    """llm_hardening_fitness 组 6 · 情绪隐蔽词扩充（独立成类）。"""

    def test_extension_words_trigger_m2(self, signal_engine):
        assert signal_engine.analyze("快疯了").mood_level == "M2"
        assert signal_engine.analyze("砸东西").mood_level == "M2"

    def test_existing_source_words_m2(self, signal_engine):
        assert signal_engine.analyze("崩溃").mood_level == "M2"
        assert signal_engine.analyze("整夜睡不着").mood_level == "M2"

    def test_m2_has_audio_resource(self, signal_engine):
        result = signal_engine.analyze("快疯了")
        assert result.mood_resources
        assert any(item.res_type == "audio" for item in result.mood_resources)

    def test_extension_word_not_m4(self, signal_engine):
        assert signal_engine.analyze("快疯了").mood_level != "M4"

    def test_s01_keywords_unchanged(self):
        from app.data.mood_tables import MOOD_SIGNALS_TABLE_TEXT

        assert "不想活了" in MOOD_SIGNALS_TABLE_TEXT
        assert "想死" in MOOD_SIGNALS_TABLE_TEXT
        assert "自杀" in MOOD_SIGNALS_TABLE_TEXT

class TestColloquialMoodWords:
    """colloquial_upgrade 组 4 · 情绪口语（独立成类）。"""

    def test_emo_maps_m1_with_resources(self, signal_engine):
        result = signal_engine.analyze("我 emo 了")
        assert result.mood_level == "M1"
        assert result.mood_resources

    def test_colloquial_words_map_m2_with_audio(self, signal_engine):
        for text in ("破防了", "摆烂了", "心塞"):
            result = signal_engine.analyze(text)
            assert result.mood_level == "M2", text
            assert any(item.res_type == "audio" for item in result.mood_resources), text

    def test_colloquial_words_never_m4(self, signal_engine):
        for text in ("emo", "破防", "摆烂", "心塞", "快疯了", "砸东西", "我心情很差"):
            assert signal_engine.analyze(text).mood_level != "M4", text

    def test_s01_priority_over_colloquial(self, signal_engine):
        assert signal_engine.analyze("摆烂，不想活了").mood_level == "M4"

    def test_mood_tables_text_unchanged(self):
        from app.data.mood_tables import MOOD_SIGNALS_TABLE_TEXT

        for keyword in ("不想活了", "想死", "自杀"):
            assert keyword in MOOD_SIGNALS_TABLE_TEXT


class TestS01PositiveNegationExemptionEngine:
    """colloquial_upgrade 组 6 · S01 正向豁免引擎级（独立成类）。"""

    def test_exempted_texts_degrade_to_m2(self, signal_engine):
        for text in ("我不想死", "不想自残，也不会自杀", "我一点也不想死"):
            result = signal_engine.analyze(text)
            assert result.mood_level == "M2", text
            assert result.mood_resources, text

    def test_mixed_texts_still_m4(self, signal_engine):
        assert signal_engine.analyze("我不想死，但我想跳楼").mood_level == "M4"
        assert signal_engine.analyze("不想死，我想割腕").mood_level == "M4"

    def test_trap_phrase_still_m4(self, signal_engine):
        assert signal_engine.analyze("不想活了").mood_level == "M4"
        assert signal_engine.analyze("快疯了，不想活了").mood_level == "M4"

    def test_existing_s01_behavior_unchanged(self, signal_engine):
        assert signal_engine.analyze("我想死").mood_level == "M4"
        assert signal_engine.analyze("我想死，笑死我了").mood_level == "M4"

    def test_exemption_with_other_signal_takes_higher(self, signal_engine):
        assert signal_engine.analyze("我不想死，但我最近很难受").mood_level == "M2"