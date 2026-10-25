"""对话 REST API：POST /api/v1/chat 与 POST /api/v1/recognize_food。

端点函数体内仅含请求校验、依赖提取与调用（禁止内联判定逻辑）；
RouterServiceError 捕获转 HTTP 500 服务错误（决策 D3 fail-closed 出口）。
recognize_food 为第 9 个端点（multipart 上传图片 → VLM 识别 → 禁忌判定）。
"""

import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field
from typing import Literal

from app.rules.agent_router import AgentRouter
from app.rules.chat_schemas import (
    CHAT_ROUTER_MISSING_MESSAGE,
    SERVICE_UNAVAILABLE_MESSAGE,
    ChatRequest,
    ChatResponse,
    FollowupContext,
    Route,
)
from app.data.memory import get_chat_history, save_chat_history
from app.rules.food_verdict import FoodVerdictService
from app.rules.rec_schemas import MealTime
from app.rules.schemas import DISCLAIMER_TEXT
from app.utils.vision import VisionClient, VisionClientError

router = APIRouter(prefix="/api/v1", tags=["chat"])

audit_logger = logging.getLogger("coremate.audit")

IMAGE_MISSING_MESSAGE = "请上传菜品图片"
IMAGE_FORMAT_MESSAGE = "请上传 jpg/png/webp 格式的图片"
IMAGE_SIZE_MESSAGE = "图片大小不能超过 5MB"
MEAL_INVALID_MESSAGE = "餐次取值不合法"
VISION_UNCONFIGURED_MESSAGE = "图像识别服务未配置，请设置 ZHIPU_API_KEY"
VISION_UNAVAILABLE_MESSAGE = "图像识别服务暂时不可用，请稍后再试"

MAX_IMAGE_BYTES = 5 * 1024 * 1024
ALLOWED_CONTENT_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})


class RecognizeFoodResponse(BaseModel):
    """菜品识别判定响应（disclaimer 模型级默认锚定，无免责的 200 不可构造）。"""

    food_name: str
    is_forbidden: bool = False
    reason: str
    swap_to: str | None = None
    disclaimer: str = Field(default=DISCLAIMER_TEXT)
    notices: list[str] = Field(default_factory=list)


def get_agent_router(request: Request) -> AgentRouter:
    """FastAPI 依赖：从应用状态提取 AgentRouter 单例（启动期装配）。"""
    agent_router = getattr(request.app.state, "agent_router", None)
    if agent_router is None:
        raise HTTPException(status_code=500, detail=CHAT_ROUTER_MISSING_MESSAGE)
    return agent_router


def get_vision_client(request: Request) -> VisionClient:
    """FastAPI 依赖：从应用状态提取视觉识别客户端单例。"""
    vision_client = getattr(request.app.state, "vision_client", None)
    if vision_client is None:
        raise HTTPException(status_code=500, detail="图像识别服务暂不可用")
    return vision_client


def get_food_verdict_service(request: Request) -> FoodVerdictService:
    """FastAPI 依赖：从应用状态提取菜品判定服务单例。"""
    verdict_service = getattr(request.app.state, "food_verdict_service", None)
    if verdict_service is None:
        raise HTTPException(status_code=500, detail="图像识别服务暂不可用")
    return verdict_service


def _append_chat_history(user_id: str, text: str, response: ChatResponse) -> None:
    """对话历史统一出口写入（裁决 D4）：M4 轮跳过 + 读-追加-写 + 静默容错。

    任何异常 → 告警日志（仅异常类型名，零对话原文）+ 当轮不入历史 +
    不向调用方抛异常（历史写入失败禁止阻断响应返回）。
    """
    try:
        if response.route == Route.M4_CRISIS:
            return
        history = get_chat_history(user_id)
        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": response.reply_text})
        save_chat_history(user_id, history)
    except Exception as exc:
        audit_logger.warning("对话历史写入异常（类型=%s），当轮不入历史", type(exc).__name__)


@router.post("/chat", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    agent_router: AgentRouter = Depends(get_agent_router),
) -> ChatResponse:
    """对话入口：user_id + text（+ 可选上轮上下文）→ 结构化响应。"""
    if not payload.user_id.strip():
        raise HTTPException(status_code=422, detail="请提供用户标识")
    if not payload.text.strip():
        raise HTTPException(status_code=422, detail="请输入对话内容")
    followup_context = FollowupContext(
        last_text=payload.last_text,
        shown_dish_ids=tuple(payload.shown_dish_ids),
        shown_action_ids=tuple(payload.shown_action_ids),
    )
    try:
        response = agent_router.route(payload.user_id, payload.text, followup_context)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=SERVICE_UNAVAILABLE_MESSAGE) from exc
    _append_chat_history(payload.user_id, payload.text, response)
    return response


@router.post("/recognize_food", response_model=RecognizeFoodResponse)
async def recognize_food(
    image: UploadFile | None = File(default=None),
    tags: list[str] = Form(default_factory=list),
    meal: str = Form(default="lunch"),
    vision_client: VisionClient = Depends(get_vision_client),
    verdict_service: FoodVerdictService = Depends(get_food_verdict_service),
) -> RecognizeFoodResponse:
    """菜品识别：上传图片 → glm-4v-flash 识别 → 健康标签禁忌判定。"""
    if image is None:
        raise HTTPException(status_code=422, detail=IMAGE_MISSING_MESSAGE)
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=422, detail=IMAGE_FORMAT_MESSAGE)

    image_bytes = await image.read()
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=422, detail=IMAGE_SIZE_MESSAGE)
    if not image_bytes:
        raise HTTPException(status_code=422, detail=IMAGE_MISSING_MESSAGE)

    try:
        meal_time = MealTime(meal)
    except ValueError:
        raise HTTPException(status_code=422, detail=MEAL_INVALID_MESSAGE) from None

    if not vision_client.is_configured():
        raise HTTPException(status_code=503, detail=VISION_UNCONFIGURED_MESSAGE)

    try:
        food_name = await vision_client.recognize_food(image_bytes, image.content_type or "image/jpeg")
    except VisionClientError as exc:
        raise HTTPException(status_code=502, detail=VISION_UNAVAILABLE_MESSAGE) from exc

    verdict = verdict_service.judge(food_name, tags, meal_time)
    return RecognizeFoodResponse(
        food_name=food_name,
        is_forbidden=verdict.is_forbidden,
        reason=verdict.reason,
        swap_to=verdict.swap_to,
        notices=verdict.notices,
    )

class HistoryEntry(BaseModel):
    """历史条目模型：role 类型域收紧为双值（模型字段即白名单）。"""

    role: Literal["user", "assistant"]
    content: str


class HistoryResponse(BaseModel):
    """历史查询/清空响应：{user_id, history, count}（空态恒非 null）。"""

    user_id: str
    history: list[HistoryEntry] = Field(default_factory=list)
    count: int = 0


def _build_history_response(user_id: str, entries: list) -> HistoryResponse:
    """共用响应构造器：count 由条目数派生（GET/DELETE 响应同构）。"""
    return HistoryResponse(user_id=user_id, history=entries, count=len(entries))


@router.get("/history/{user_id}", response_model=HistoryResponse)
def get_history(user_id: str) -> HistoryResponse:
    """历史查询：只读直连存储，时间正序（最早在前），≤20 由窗口约束。"""
    if not user_id.strip():
        raise HTTPException(status_code=422, detail="请提供用户标识")
    entries = get_chat_history(user_id)
    return _build_history_response(user_id, entries)


@router.delete("/history/{user_id}", response_model=HistoryResponse)
def delete_history(user_id: str) -> HistoryResponse:
    """历史清空：覆写空列表（只清 history，三类偏好保留）；幂等。"""
    if not user_id.strip():
        raise HTTPException(status_code=422, detail="请提供用户标识")
    if not save_chat_history(user_id, []):
        raise HTTPException(status_code=500, detail=SERVICE_UNAVAILABLE_MESSAGE)
    return _build_history_response(user_id, [])