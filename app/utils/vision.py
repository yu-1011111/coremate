"""智谱视觉真实客户端：菜品图像识别（real_api_integration spec 5.3）。

凭证经 os.getenv 占位默认值读取（未配置态由端点转 503）；recognize_food
为异步调用（glm-4v-flash），请求体仅含 base64 图片与冻结 Prompt，
禁止携带任何用户数据（tags / user_id 不出境，spec 4.3 数据最小化）。
"""

import base64
import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger("coremate.data")

ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "YOUR_ZHIPU_KEY_HERE")
ZHIPU_CHAT_COMPLETIONS_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
VISION_MODEL = "glm-4v-flash"
VISION_TIMEOUT_SECONDS = 15.0
FOOD_RECOGNITION_PROMPT = (
    "你是一个食堂菜品识别助手。请识别图片中的食物，只回答食物名称，如：麻辣香锅。"
    "如果无法识别，返回'未知菜品'。"
)

_KEY_PLACEHOLDER = "YOUR_ZHIPU_KEY_HERE"


class VisionClientError(Exception):
    """视觉识别调用失败（网络/超时/非 2xx/结构异常）。"""


def is_configured() -> bool:
    """单变量判定：Key 非空且非占位值。"""
    return bool(ZHIPU_API_KEY) and ZHIPU_API_KEY != _KEY_PLACEHOLDER


class VisionClient:
    """智谱 glm-4v-flash 菜品识别客户端（异步短生命周期）。"""

    def __init__(self, transport: Optional[httpx.AsyncBaseTransport] = None) -> None:
        self._transport = transport

    @staticmethod
    def is_configured() -> bool:
        """单变量判定：Key 非空且非占位值（委托模块级判定）。"""
        return is_configured()

    async def recognize_food(self, image_bytes: bytes, mime_type: str) -> str:
        """识别图片中的菜名：返回 VLM 回复文本（去首尾空白）。

        失败一律抛 VisionClientError；"未知菜品"是 VLM 正常业务返回，
        与调用失败可区分（由判定服务分支处理）。
        """
        if not image_bytes:
            raise VisionClientError("图片内容为空")

        encoded = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:{mime_type};base64,{encoded}"
        payload = {
            "model": VISION_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": FOOD_RECOGNITION_PROMPT},
                    ],
                }
            ],
        }
        try:
            async with httpx.AsyncClient(
                headers={"Authorization": f"Bearer {ZHIPU_API_KEY}"},
                timeout=VISION_TIMEOUT_SECONDS,
                transport=self._transport,
            ) as client:
                response = await client.post(ZHIPU_CHAT_COMPLETIONS_URL, json=payload)
        except Exception as exc:
            logger.warning("视觉识别调用异常（类型=%s）", type(exc).__name__)
            raise VisionClientError("视觉识别调用失败") from exc

        if response.status_code != 200:
            logger.warning("视觉识别非 2xx 响应: %s", response.status_code)
            raise VisionClientError("视觉识别服务响应异常")
        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            logger.warning("视觉识别响应结构异常")
            raise VisionClientError("视觉识别响应结构异常") from exc
        if not isinstance(content, str):
            raise VisionClientError("视觉识别回复非文本")
        return content.strip()