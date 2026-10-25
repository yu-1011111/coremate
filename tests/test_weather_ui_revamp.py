"""天气联动修复与 UI 阿福蓝改版验收测试（weather_ui_revamp tasks.md 任务组 7）。

8 个测试类：强制触发 / 温度透传 / 状态句 / 室内切换 / 静默缺省 /
自然引导 / UI 蓝锚 / 退役常量守护。全部用 fake WeatherClient 替身
（不依赖真实凭证与网络）；新增用例不计入 339 红线基数。
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.rules.agent_router import create_default_agent_router
from app.rules.intent_parser import IntentParser
from app.utils.weather import (
    WeatherQueryResult,
    build_weather_status_sentence,
)


class _FakeWeatherClient:
    """测试替身：实现 WeatherClientLike 协议。"""

    def __init__(self, result=None, raises=False):
        self._result = result
        self._raises = raises
        self.call_count = 0

    def is_configured(self) -> bool:
        return True

    def get_weather_now(self) -> WeatherQueryResult:
        self.call_count += 1
        if self._raises:
            raise RuntimeError("weather broken")
        return self._result


def _router_with(weather_client):
    router = create_default_agent_router()
    router._weather_client = weather_client
    return router


def _rainy(temp="22"):
    return _FakeWeatherClient(WeatherQueryResult(status="success", weather="雨", raw_text="小雨", temp=temp))


def _sunny(temp=None):
    return _FakeWeatherClient(WeatherQueryResult(status="success", weather="晴", raw_text="晴", temp=temp))


class TestForcedWeatherTrigger:
    """7.1a 强制触发并集锚点：运动意图 OR 室外触发词。"""

    def test_exercise_intent_with_trigger_word_calls_once(self):
        fake = _rainy()
        response = _router_with(fake).route("u1", "我想去操场跑步，有什么建议？")
        assert response.route.value == "HEALTH_QUERY"
        assert fake.call_count == 1

    def test_diet_only_with_trigger_word_calls_once(self):
        fake = _rainy()
        response = _router_with(fake).route("u1", "操场边上吃点什么好？")
        assert response.route.value == "HEALTH_QUERY"
        assert fake.call_count == 1

    def test_diet_only_without_trigger_skips(self):
        fake = _rainy()
        response = _router_with(fake).route("u1", "尿酸高，午饭吃什么？")
        assert response.route.value == "HEALTH_QUERY"
        assert fake.call_count == 0

    def test_mood_with_exercise_intent_calls_once(self):
        fake = _rainy()
        response = _router_with(fake).route("u1", "我最近很焦虑，在操场怎么运动？")
        assert response.route.value == "MOOD_SUPPORT"
        assert fake.call_count == 1

    def test_small_talk_skips(self):
        fake = _rainy()
        response = _router_with(fake).route("u1", "今天天气不错")
        assert response.route.value == "SMALL_TALK"
        assert fake.call_count == 0

    def test_crisis_skips(self):
        fake = _rainy()
        response = _router_with(fake).route("u1", "我想死")
        assert response.route.value == "M4_CRISIS"
        assert fake.call_count == 0
        assert response.crisis is not None


class TestWeatherTempPassthrough:
    """7.1b 温度透传锚点。"""

    def test_success_carries_temp(self):
        result = WeatherQueryResult(status="success", weather="雨", raw_text="小雨", temp="22")
        assert result.temp == "22"
        assert result.raw_text == "小雨"

    def test_non_success_temp_is_none(self):
        assert WeatherQueryResult(status="unconfigured").temp is None
        assert WeatherQueryResult(status="failed").temp is None


class TestWeatherStatusSentence:
    """7.2a 天气状态句锚点。"""

    def test_status_sentence_at_reply_start(self):
        response = _router_with(_rainy("22")).route("u1", "我想去操场跑步，有什么建议？")
        assert response.reply_text.startswith("徐州今日小雨，22度。")

    def test_status_sentence_uses_raw_text(self):
        fake = _FakeWeatherClient(
            WeatherQueryResult(status="success", weather="雨", raw_text="中到大雨", temp="18")
        )
        response = _router_with(fake).route("u1", "我想去操场跑步，有什么建议？")
        assert response.reply_text.startswith("徐州今日中到大雨，18度。")

    def test_status_sentence_without_mapping_still_prefixed(self):
        fake = _FakeWeatherClient(
            WeatherQueryResult(status="success", weather=None, raw_text="多云", temp="20")
        )
        response = _router_with(fake).route("u1", "我想去操场跑步，有什么建议？")
        assert response.reply_text.startswith("徐州今日多云，20度。")

    def test_temp_not_numeric_degrades(self):
        assert build_weather_status_sentence(
            WeatherQueryResult(status="success", weather="雨", raw_text="中到大雨", temp="abc")
        ) == "徐州今日中到大雨。"

    def test_failed_state_has_no_status_sentence(self):
        fake = _FakeWeatherClient(WeatherQueryResult(status="failed"))
        response = _router_with(fake).route("u1", "我想去操场跑步，有什么建议？")
        assert "徐州今日" not in response.reply_text
        assert "天气服务暂不可用，已按当前条件推荐" in response.notices

    def test_mood_route_sothing_script_first(self):
        response = _router_with(_rainy()).route("u1", "我最近很焦虑，在操场怎么运动？")
        assert response.reply_text.startswith("听起来你最近过得不太轻松")


class TestIndoorSwitch:
    """7.2b 室内切换锚点。"""

    def test_rain_playground_switches_to_indoor(self):
        response = _router_with(_rainy()).route("u1", "我想去操场跑步，有什么建议？")
        actions = response.exercise_recommendations.recommended_actions
        assert actions
        assert response.exercise_recommendations.run_courses == []
        assert "已为您查询天气，当前下雨，已自动为您切换为室内方案" in response.reply_text

    def test_sunny_playground_keeps_outdoor(self):
        response = _router_with(_sunny()).route("u1", "我想去操场跑步，有什么建议？")
        assert "已自动为您切换为室内方案" not in response.reply_text
        assert response.exercise_recommendations.recommended_actions

    def test_cloudy_playground_no_switch(self):
        fake = _FakeWeatherClient(
            WeatherQueryResult(status="success", weather=None, raw_text="多云", temp="20")
        )
        response = _router_with(fake).route("u1", "我想去操场跑步，有什么建议？")
        assert "已自动为您切换为室内方案" not in response.reply_text

    def test_test_crash_rain_keeps_course_degradation(self):
        response = _router_with(_rainy()).route("u1", "体测要及格，操场怎么练？")
        courses = response.exercise_recommendations.run_courses
        assert "已为您查询天气，当前下雨，已自动为您切换为室内方案" not in response.reply_text

    def test_dorm_rain_no_swap_sentence(self):
        response = _router_with(_rainy()).route("u1", "在宿舍怎么运动？")
        assert "已自动为您切换为室内方案" not in response.reply_text

    def test_failed_state_keeps_playground_with_text_weather(self):
        fake = _FakeWeatherClient(WeatherQueryResult(status="failed"))
        response = _router_with(fake).route("u1", "下雨了，在操场怎么运动 30 分钟？")
        r001 = next(c for c in response.exercise_recommendations.run_courses if c.course_id == "R001")
        assert r001.weather_blocked is True

    def test_mood_rain_playground_slim_indoor(self):
        response = _router_with(_rainy()).route("u1", "我最近很焦虑，在操场怎么运动？")
        assert response.route.value == "MOOD_SUPPORT"
        assert "已自动为您切换为室内方案" in response.reply_text
        assert response.reply_text.rstrip().endswith("已自动为您切换为室内方案")


class TestSilentDefaultMinutes:
    """7.3a 静默缺省锚点。"""

    def test_default_15_silent(self):
        result = IntentParser().parse("怎么运动？")
        assert result.available_minutes == 15
        assert "未识别运动时长" not in "".join(result.notices)
        assert "未识别运动场景，已默认按操场推荐" in result.notices

    def test_explicit_15_not_defaulted(self):
        result = IntentParser().parse("给我安排 15 分钟的运动")
        assert result.available_minutes == 15
        assert result.minutes_defaulted is False

    def test_default_flag_true_when_missing(self):
        result = IntentParser().parse("怎么运动？")
        assert result.minutes_defaulted is True

    def test_explicit_30_extracted(self):
        result = IntentParser().parse("给我安排 30 分钟的运动")
        assert result.available_minutes == 30
        assert result.minutes_defaulted is False

    def test_out_of_range_silently_defaults(self):
        result = IntentParser().parse("帮我安排 500 分钟的运动")
        assert result.available_minutes == 15
        assert result.minutes_defaulted is True


class TestNaturalGuideScript:
    """7.3b 自然引导话术锚点。"""

    def test_pure_exercise_default_minutes_uses_natural_script(self):
        response = _router_with(_sunny()).route("u1", "我想去操场跑步，有什么建议？")
        assert "为你推荐 15 分钟的微运动方案：" in response.reply_text

    def test_explicit_minutes_uses_generic_script(self):
        response = _router_with(_sunny()).route("u1", "在操场怎么运动 30 分钟？")
        assert "为你推荐 15 分钟的微运动方案：" not in response.reply_text
        assert "为你整理了" in response.reply_text

    def test_dual_intent_uses_generic_script(self):
        response = _router_with(_sunny()).route("u1", "午饭吃什么？顺便在操场怎么运动？")
        assert "为你整理了" in response.reply_text
        assert "未识别运动时长" not in "".join(response.notices)

    def test_mood_default_minutes_keeps_slim_format(self):
        response = _router_with(_sunny()).route("u1", "我最近很焦虑，在操场怎么运动？")
        assert response.reply_text.startswith("听起来你最近过得不太轻松")
        assert "想动一动的话" in response.reply_text


@pytest.fixture(scope="module")
def index_html():
    client = TestClient(create_app())
    response = client.get("/")
    assert response.status_code == 200
    return response.text


class TestUiAfBlueTheme:
    """7.4a UI 阿福蓝锚点（GET / 源码断言）。"""

    def test_blue_600_anchors_present(self, index_html):
        assert "from-blue-600 to-blue-500" in index_html
        assert "bg-blue-600 text-white rounded-2xl" in index_html
        assert "bg-blue-600 text-white flex items-center justify-center hover:bg-blue-700" in index_html

    def test_teal_indigo_zero_leftover(self, index_html):
        assert "teal" not in index_html
        assert "indigo" not in index_html

    def test_page_background_slate_50(self, index_html):
        assert "bg-slate-50 h-screen" in index_html

    def test_recommend_card_left_border(self, index_html):
        assert "border-l-4 border-emerald-400 bg-emerald-50" in index_html
        assert "border-l-4 border-red-300 bg-red-50" in index_html

    def test_red_safety_anchors_kept(self, index_html):
        assert "border-red-500" in index_html
        assert "border-red-300" in index_html

    def test_functional_colors_kept(self, index_html):
        assert "violet" in index_html
        assert "bg-emerald-100 text-emerald-700" in index_html

    def test_blue_focus_ring_and_icons(self, index_html):
        assert "focus:ring-blue-400" in index_html
        assert "text-blue-600 text-xl" in index_html
        assert "has-[:checked]:border-blue-400" in index_html


class TestRetiredConstants:
    """7.4b 退役常量守护锚点。"""

    def test_minutes_notice_retired(self):
        for path in (Path("app"), Path("app/templates")):
            for source in path.rglob("*.py"):
                assert "未识别运动时长" not in source.read_text(encoding="utf-8")
        html = Path("app/templates/index.html").read_text(encoding="utf-8")
        assert "未识别运动时长" not in html

    def test_old_weather_template_retired(self):
        weather_source = Path("app/utils/weather.py").read_text(encoding="utf-8")
        assert "WEATHER_SWAP_REPLY_TEMPLATE" not in weather_source
        assert "已查询徐州今日天气" not in weather_source

    def test_script_minutes_single_source(self):
        from app.rules.chat_schemas import EXERCISE_DEFAULT_SCRIPT
        from app.rules.intent_parser import DEFAULT_MINUTES

        assert "15" in EXERCISE_DEFAULT_SCRIPT
        assert DEFAULT_MINUTES == 15