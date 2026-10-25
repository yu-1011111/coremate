"""和风天气真实客户端：现时天气查询（real_api_integration spec 5.2）。

凭证经 os.getenv 占位默认值读取（未配置态零网络开销，静默跳过）；
WeatherClient 为同步短生命周期调用（设计决策 D1：route() 保持同步），
查询结果三态归约（unconfigured / success / failed），方法不抛任何异常。
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

import httpx

from app.rules.rec_schemas import WEATHER_VALUES

logger = logging.getLogger("coremate.data")

QWEATHER_API_KEY = os.getenv("QWEATHER_API_KEY", "YOUR_QWEATHER_KEY_HERE")
QWEATHER_API_HOST = os.getenv("QWEATHER_API_HOST", "YOUR_QWEATHER_HOST_HERE")

WEATHER_TIMEOUT_SECONDS = 3.0
QWEATHER_LOCATION_ID = "101190801"

WEATHER_TEXT_MAP: tuple[tuple[str, str], ...] = (
    ("雨", "雨"),
    ("雪", "雪"),
    ("霾", "霾"),
    ("晴", "晴"),
)
WEATHER_TEMP_HIGH_THRESHOLD = 32
WEATHER_TEMP_LOW_THRESHOLD = 5

WEATHER_UNAVAILABLE_NOTICE = "天气服务暂不可用，已按当前条件推荐"
WEATHER_STATUS_TEMPLATE = "徐州今日{weather_text}，{temp}度。"
WEATHER_STATUS_TEMPLATE_NO_TEMP = "徐州今日{weather_text}。"
WEATHER_INDOOR_SWAP_TEMPLATE = "已为您查询天气，当前{现象表述}，已自动为您切换为室内方案"
INDOOR_WEATHER_DESCRIPTIONS: dict[str, str] = {
    "雨": "下雨",
    "雪": "下雪",
    "霾": "有雾霾",
}

_UNKNOWN_FOOD_PLACEHOLDERS = ("YOUR_QWEATHER_KEY_HERE", "YOUR_QWEATHER_HOST_HERE")


@dataclass(frozen=True)
class WeatherQueryResult:
    """天气查询三态结果：status ∈ unconfigured / success / failed。

    success 态携带 weather（归一化枚举值）/ raw_text（API 原文现象）/
    temp（气温字符串）三元数据；失败与未配置态不携带。
    """

    status: str
    weather: Optional[str] = None
    raw_text: Optional[str] = None
    temp: Optional[str] = None


def is_configured() -> bool:
    """双变量完备判定：Key 与 Host 均非空且均非占位值才为已配置态。"""
    return (
        bool(QWEATHER_API_KEY)
        and QWEATHER_API_KEY not in _UNKNOWN_FOOD_PLACEHOLDERS
        and bool(QWEATHER_API_HOST)
        and QWEATHER_API_HOST not in _UNKNOWN_FOOD_PLACEHOLDERS
    )


def normalize_weather(text: str, temp: str) -> Optional[str]:
    """天气文本归一化纯函数：映射到既有运动引擎天气枚举。

    判定优先级：雨/雪/霾（text 含对应字）> 晴 > 高温（temp≥32）> 低温（temp≤5）
    > 无天气（None）；输出取值锚定 WEATHER_VALUES 冻结枚举，禁止新增枚举值。
    """
    for keyword, weather in WEATHER_TEXT_MAP:
        if keyword in text:
            return weather
    try:
        temperature = float(temp)
    except (TypeError, ValueError):
        return None
    if temperature >= WEATHER_TEMP_HIGH_THRESHOLD:
        return "高温"
    if temperature <= WEATHER_TEMP_LOW_THRESHOLD:
        return "低温"
    return None


_ALLOWED_WEATHER_VALUES = frozenset(WEATHER_VALUES)
for _, mapped in WEATHER_TEXT_MAP:
    assert mapped in _ALLOWED_WEATHER_VALUES, f"归一化映射输出越界: {mapped}"


def build_weather_status_sentence(result: WeatherQueryResult) -> Optional[str]:
    """组装天气状态句：成功态以 API 原文现象播报（归一化值禁止进入）。

    temp 可解析为数字 → 完整模板"徐州今日{原文}，{temp}度。"；
    temp 缺失/非数字 → 退化模板"徐州今日{原文}。"；非成功态返回 None。
    """
    if result.status != "success" or not result.raw_text:
        return None
    try:
        float(result.temp or "")
    except (TypeError, ValueError):
        return WEATHER_STATUS_TEMPLATE_NO_TEMP.format(weather_text=result.raw_text)
    return WEATHER_STATUS_TEMPLATE.format(weather_text=result.raw_text, temp=result.temp)


class WeatherClient:
    """和风天气现时天气客户端（同步短生命周期，无实例可变状态）。"""

    def __init__(self, transport: Optional[httpx.BaseTransport] = None) -> None:
        self._transport = transport

    @staticmethod
    def is_configured() -> bool:
        """双变量完备判定（委托模块级判定）。"""
        return is_configured()

    def get_weather_now(self) -> WeatherQueryResult:
        """查询徐州现时天气：失败一律归约为 failed 态，不抛异常。"""
        if not is_configured():
            return WeatherQueryResult(status="unconfigured")

        url = f"https://{QWEATHER_API_HOST}/v7/weather/now?location={QWEATHER_LOCATION_ID}"
        try:
            with httpx.Client(
                headers={"X-QW-Api-Key": QWEATHER_API_KEY},
                timeout=WEATHER_TIMEOUT_SECONDS,
                transport=self._transport,
            ) as client:
                response = client.get(url)
            if response.status_code != 200:
                logger.warning("天气查询非 2xx 响应: %s", response.status_code)
                return WeatherQueryResult(status="failed")
            payload = response.json()
            now = payload.get("now") or {}
            text = now.get("text")
            temp = now.get("temp")
            if not isinstance(text, str) or not isinstance(temp, str):
                logger.warning("天气响应结构异常，缺少 now.text / now.temp")
                return WeatherQueryResult(status="failed")
            return WeatherQueryResult(
                status="success",
                weather=normalize_weather(text, temp),
                raw_text=text,
                temp=temp,
            )
        except Exception as exc:
            logger.warning("天气查询异常（类型=%s）", type(exc).__name__)
            return WeatherQueryResult(status="failed")