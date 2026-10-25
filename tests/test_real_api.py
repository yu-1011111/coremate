"""真实外部 API 接入验收测试（tasks.md 任务组 10）。

覆盖：WeatherClient 单测（MockTransport）、编排天气联动、recognize_food
端点全分支、双实现一致性锚定与依赖基线断言。全部用 pytest + httpx
MockTransport + TestClient + monkeypatch，不引入新测试依赖（决策 D11）；
无 Key 环境下既有 300 用例红线由结构性保证（另见 10.6 全量回归）。
"""

import pytest
import httpx
from fastapi.testclient import TestClient

import app.utils.vision as vision_module
import app.utils.weather as weather_module
from app.main import create_app
from app.rules.chat_schemas import DISCLAIMER_TEXT
from app.rules.diet import create_default_diet_engine
from app.rules.food_verdict import (
    FoodVerdictService,
    UNKNOWN_FOOD_TEXT,
    create_default_food_verdict_service,
)
from app.rules.rec_schemas import MealTime
from app.utils.vision import VisionClient
from app.utils.weather import WeatherClient, WeatherQueryResult, normalize_weather


def _configure_weather(monkeypatch):
    """将天气模块常量替换为已配置态（规避导入期取值陷阱）。"""
    monkeypatch.setattr(weather_module, "QWEATHER_API_KEY", "test-key-123")
    monkeypatch.setattr(weather_module, "QWEATHER_API_HOST", "devapi.qweather.com")


class TestWeatherClient:
    """10.1 WeatherClient 客户端单测。"""


    def test_unconfigured_makes_zero_requests(self, monkeypatch):
        """受控未配置态：凭证常量替换回占位值后零网络请求（环境无关）。"""
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json={"now": {"text": "晴", "temp": "20"}})

        monkeypatch.setattr(weather_module, "QWEATHER_API_KEY", "YOUR_QWEATHER_KEY_HERE")
        monkeypatch.setattr(weather_module, "QWEATHER_API_HOST", "YOUR_QWEATHER_HOST_HERE")
        client = WeatherClient(transport=httpx.MockTransport(handler))
        assert weather_module.is_configured() is False
        result = client.get_weather_now()
        assert result.status == "unconfigured"
        assert requests == []

    def test_partial_credentials_still_unconfigured(self, monkeypatch):
        _configure_weather(monkeypatch)
        monkeypatch.setattr(weather_module, "QWEATHER_API_HOST", "YOUR_QWEATHER_HOST_HERE")
        assert weather_module.is_configured() is False

    def test_request_url_and_header(self, monkeypatch):
        _configure_weather(monkeypatch)
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["key"] = request.headers.get("X-QW-Api-Key")
            return httpx.Response(200, json={"now": {"text": "小雨", "temp": "18"}})

        result = WeatherClient(transport=httpx.MockTransport(handler)).get_weather_now()
        assert captured["url"] == "https://devapi.qweather.com/v7/weather/now?location=101190801"
        assert captured["key"] == "test-key-123"
        assert result.status == "success"
        assert result.weather == "雨"
        assert result.raw_text == "小雨"

    @pytest.mark.parametrize(
        "text,temp,expected",
        [
            ("小雨", "18", "雨"),
            ("多云", "20", None),
            ("晴", "35", "晴"),
            ("小雨", "35", "雨"),
            ("晴", "3", "晴"),
            ("多云", "35", "高温"),
            ("多云", "3", "低温"),
            ("中雪", "-2", "雪"),
            ("霾", "20", "霾"),
        ],
    )
    def test_normalize_weather_mapping(self, text, temp, expected):
        assert normalize_weather(text, temp) == expected

    @pytest.mark.parametrize(
        "status_code,payload",
        [
            (500, {"now": {"text": "晴", "temp": "20"}}),
            (200, {"now": {}}),
            (200, {}),
            (200, {"now": {"text": "晴"}}),
        ],
    )
    def test_failures_reduce_to_failed(self, monkeypatch, status_code, payload):
        _configure_weather(monkeypatch)
        client = WeatherClient(transport=httpx.MockTransport(lambda req: httpx.Response(status_code, json=payload)))
        assert client.get_weather_now().status == "failed"

    def test_network_exception_reduces_to_failed(self, monkeypatch):
        _configure_weather(monkeypatch)

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("boom")

        client = WeatherClient(transport=httpx.MockTransport(handler))
        assert client.get_weather_now().status == "failed"


class _FakeWeatherClient:
    """测试替身：实现 WeatherClientLike 协议。"""

    def __init__(self, result=None, raises=False, configured=True):
        self._result = result
        self._raises = raises
        self._configured = configured
        self.call_count = 0

    def is_configured(self) -> bool:
        return self._configured

    def get_weather_now(self) -> WeatherQueryResult:
        self.call_count += 1
        if self._raises:
            raise RuntimeError("weather broken")
        return self._result


class TestRouterWeather:
    """10.2 编排层天气联动测试。"""

    def _router_with(self, weather_client):
        from app.rules.agent_router import create_default_agent_router

        router = create_default_agent_router()
        router._weather_client = weather_client
        return router

    def test_rain_weather_switches_to_indoor(self):
        fake = _FakeWeatherClient(WeatherQueryResult(status="success", weather="雨", raw_text="小雨", temp="22"))
        response = self._router_with(fake).route("u1", "我想去操场跑步，有什么建议？")
        assert response.route.value == "HEALTH_QUERY"
        assert response.reply_text.startswith("徐州今日小雨，22度。")
        assert "已为您查询天气，当前下雨，已自动为您切换为室内方案" in response.reply_text
        assert response.exercise_recommendations.run_courses == []
        assert response.exercise_recommendations.recommended_actions

    def test_sunny_weather_no_notice(self):
        fake = _FakeWeatherClient(WeatherQueryResult(status="success", weather="晴", raw_text="晴"))
        response = self._router_with(fake).route("u1", "我想去操场跑步，有什么建议？")
        assert response.reply_text.startswith("徐州今日晴")
        assert "已自动为您切换为室内方案" not in response.reply_text

    def test_failed_weather_falls_back_to_text_extraction(self):
        fake = _FakeWeatherClient(WeatherQueryResult(status="failed"))
        response = self._router_with(fake).route("u1", "下雨了，在操场怎么运动 30 分钟？")
        assert "天气服务暂不可用，已按当前条件推荐" in response.notices
        r001 = next(c for c in response.exercise_recommendations.run_courses if c.course_id == "R001")
        assert r001.weather_blocked is True

    def test_crisis_never_calls_weather(self):
        fake = _FakeWeatherClient(WeatherQueryResult(status="success", weather="雨", raw_text="小雨"))
        response = self._router_with(fake).route("u1", "我想死")
        assert response.route.value == "M4_CRISIS"
        assert fake.call_count == 0

    def test_raising_client_degrades_gracefully(self):
        fake = _FakeWeatherClient(raises=True)
        response = self._router_with(fake).route("u1", "在操场怎么运动半小时？")
        assert response.route.value == "HEALTH_QUERY"
        assert "天气服务暂不可用，已按当前条件推荐" in response.notices

    def test_pure_diet_intent_skips_weather(self):
        fake = _FakeWeatherClient(WeatherQueryResult(status="success", weather="雨", raw_text="小雨"))
        response = self._router_with(fake).route("u1", "尿酸高，午饭吃什么？")
        assert response.route.value == "HEALTH_QUERY"
        assert fake.call_count == 0


def _vision_transport(reply_content: str):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": reply_content}}]},
        )

    return httpx.MockTransport(handler)


@pytest.fixture()
def configured_vision_app(monkeypatch):
    monkeypatch.setattr(vision_module, "ZHIPU_API_KEY", "test-zhipu-key")
    application = create_app()
    application.state.vision_client = VisionClient(transport=_vision_transport(" 麻辣香锅 "))
    return TestClient(application)


class TestRecognizeFoodEndpoint:
    """10.3 recognize_food 端点测试。"""

    def test_success_with_tags_forbidden(self, configured_vision_app):
        response = configured_vision_app.post(
            "/api/v1/recognize_food",
            files={"image": ("food.jpg", b"\xff\xd8\xff\xe0fake-jpg", "image/jpeg")},
            data={"tags": ["超重"], "meal": "lunch"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["food_name"] == "麻辣香锅"
        assert body["is_forbidden"] is True
        assert body["reason"]
        assert body["swap_to"]
        assert body["disclaimer"] == DISCLAIMER_TEXT

    def test_unknown_food_branch(self, monkeypatch):
        monkeypatch.setattr(vision_module, "ZHIPU_API_KEY", "test-zhipu-key")
        application = create_app()
        application.state.vision_client = VisionClient(transport=_vision_transport(UNKNOWN_FOOD_TEXT))
        client = TestClient(application)
        response = client.post(
            "/api/v1/recognize_food",
            files={"image": ("food.jpg", b"fake", "image/jpeg")},
            data={"tags": ["高尿酸"]},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["food_name"] == UNKNOWN_FOOD_TEXT
        assert body["is_forbidden"] is False
        assert "未能识别" in body["reason"]

    def test_contains_match(self, monkeypatch):
        monkeypatch.setattr(vision_module, "ZHIPU_API_KEY", "test-zhipu-key")
        application = create_app()
        application.state.vision_client = VisionClient(transport=_vision_transport("香锅"))
        client = TestClient(application)
        response = client.post(
            "/api/v1/recognize_food",
            files={"image": ("food.jpg", b"fake", "image/jpeg")},
            data={"tags": ["高尿酸"]},
        )
        assert response.status_code == 200
        body = response.json()
        assert "香锅" in body["food_name"] or body["food_name"] == "香锅"

    def test_dish_not_in_catalog(self, monkeypatch):
        monkeypatch.setattr(vision_module, "ZHIPU_API_KEY", "test-zhipu-key")
        application = create_app()
        application.state.vision_client = VisionClient(transport=_vision_transport("外星料理"))
        client = TestClient(application)
        response = client.post(
            "/api/v1/recognize_food",
            files={"image": ("food.jpg", b"fake", "image/jpeg")},
            data={"tags": ["高尿酸"]},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["is_forbidden"] is False
        assert "暂未收录" in body["reason"]

    def test_no_tags_universal_verdict(self, configured_vision_app):
        response = configured_vision_app.post(
            "/api/v1/recognize_food",
            files={"image": ("food.jpg", b"fake", "image/jpeg")},
        )
        assert response.status_code == 200
        assert response.json()["is_forbidden"] is False

    def test_unknown_tag_ignored(self, configured_vision_app):
        response = configured_vision_app.post(
            "/api/v1/recognize_food",
            files={"image": ("food.jpg", b"fake", "image/jpeg")},
            data={"tags": ["超重", "外星体质"]},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["is_forbidden"] is True
        assert "存在未识别的健康标签" in " ".join(body["notices"])

    def test_gif_rejected_422(self, configured_vision_app):
        response = configured_vision_app.post(
            "/api/v1/recognize_food",
            files={"image": ("a.gif", b"fake", "image/gif")},
        )
        assert response.status_code == 422
        assert "jpg/png/webp" in str(response.json()["detail"])

    def test_oversize_image_422(self, configured_vision_app):
        big = b"x" * (5 * 1024 * 1024 + 1)
        response = configured_vision_app.post(
            "/api/v1/recognize_food",
            files={"image": ("big.png", big, "image/png")},
        )
        assert response.status_code == 422
        assert "5MB" in str(response.json()["detail"])

    def test_image_missing_422(self, configured_vision_app):
        response = configured_vision_app.post("/api/v1/recognize_food", data={})
        assert response.status_code == 422

    def test_unconfigured_key_503(self, monkeypatch):
        """受控未配置态：凭证常量替换回占位值后端点返回 503（环境无关）。"""
        monkeypatch.setattr(vision_module, "ZHIPU_API_KEY", "YOUR_ZHIPU_KEY_HERE")
        client = TestClient(create_app())
        response = client.post(
            "/api/v1/recognize_food",
            files={"image": ("a.jpg", b"fake", "image/jpeg")},
        )
        assert response.status_code == 503
        assert "ZHIPU_API_KEY" in str(response.json()["detail"])

    def test_vision_failure_502(self, monkeypatch):
        monkeypatch.setattr(vision_module, "ZHIPU_API_KEY", "test-zhipu-key")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": "boom"})

        application = create_app()
        application.state.vision_client = VisionClient(transport=httpx.MockTransport(handler))
        client = TestClient(application)
        response = client.post(
            "/api/v1/recognize_food",
            files={"image": ("a.jpg", b"fake", "image/jpeg")},
        )
        assert response.status_code == 502
        assert "图像识别服务暂时不可用" in str(response.json()["detail"])

    def test_vision_request_carries_no_user_data(self, monkeypatch):
        monkeypatch.setattr(vision_module, "ZHIPU_API_KEY", "test-zhipu-key")
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = request.content.decode("utf-8")
            return httpx.Response(200, json={"choices": [{"message": {"content": "麻辣香锅"}}]})

        application = create_app()
        application.state.vision_client = VisionClient(transport=httpx.MockTransport(handler))
        client = TestClient(application)
        client.post(
            "/api/v1/recognize_food",
            files={"image": ("food.jpg", b"fakeimage", "image/jpeg")},
            data={"tags": ["高尿酸", "超重"], "meal": "lunch"},
        )
        body = captured["body"]
        assert "高尿酸" not in body
        assert "超重" not in body
        assert "base64" in body
        assert "glm-4v-flash" in body


class TestConsistencyAnchors:
    """10.4 双实现一致性锚定与依赖基线断言。"""

    def test_food_verdict_matches_diet_engine_reason(self):
        verdict_service = create_default_food_verdict_service()
        diet_engine = create_default_diet_engine()
        catalog_dish = next(
            d for d in verdict_service._catalog.dishes if d.name == "猪肝"
        )
        tags = ["高尿酸"]
        verdict = verdict_service.judge("猪肝", tags, MealTime.LUNCH)
        diet_result = diet_engine.recommend(tags, MealTime.LUNCH)
        avoided = next(d for d in diet_result.avoided_dishes if d.dish_id == catalog_dish.dish_id)
        assert verdict.is_forbidden is True
        assert verdict.reason == avoided.reason

    def test_requirements_baseline(self):
        from pathlib import Path

        content = Path("requirements.txt").read_text(encoding="utf-8")
        lines = [line for line in content.splitlines() if line.strip()]
        assert len(lines) == 8
        assert any(line.strip() == "python-multipart>=0.0.9" for line in lines)
        assert any(line.strip() == "python-dotenv>=1.0.0" for line in lines)
        for line in lines:
            assert line.count(">=") == 1, f"行粘连或格式异常: {line}"

    def test_engines_have_no_httpx_import(self):
        from pathlib import Path

        for filename in ("exercise.py", "diet.py"):
            source = Path("app/rules") / filename
            assert "httpx" not in source.read_text(encoding="utf-8")

    def test_no_real_credentials_in_source(self):
        """凭证形态锚点：三客户端模块存在凭证常量定义行（配置方式 A/B 均合法）。"""
        from pathlib import Path

        weather_source = Path("app/utils/weather.py").read_text(encoding="utf-8")
        vision_source = Path("app/utils/vision.py").read_text(encoding="utf-8")
        assert "QWEATHER_API_KEY = " in weather_source
        assert "QWEATHER_API_HOST = " in weather_source
        assert "ZHIPU_API_KEY = " in vision_source
        for path in Path("app").rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            assert "X-QW-Api-Key: sk-" not in source
            break