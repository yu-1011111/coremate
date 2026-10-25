"""芯动管家 CoreMate 应用入口：FastAPI 应用工厂与装配。

装配内容：注册健康分析路由、启动期构建引擎单例、挂载静态资源目录、
注册档案级校验错误的中文提示转换、配置审计日志文件输出。
"""

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from dotenv import load_dotenv

load_dotenv()

from app.api.chat import router as chat_router
from app.api.health_analysis import router as health_router
from app.api.mood import router as mood_router
from app.api.pages import router as pages_router
from app.api.recommend import router as recommend_router
from app.rules.agent_router import create_default_agent_router
from app.rules.crisis import create_default_crisis_responder
from app.rules.diet import create_default_diet_engine
from app.rules.exercise import create_default_exercise_engine
from app.rules.food_verdict import create_default_food_verdict_service
from app.rules.medical import create_default_engine
from app.rules.selfcheck import create_default_selfcheck_engine
from app.rules.signal_engine import create_default_signal_engine
from app.rules.sports_course import create_default_sports_course_advisor
from app.utils.llm import LLMClient
from app.utils.vision import VisionClient
from app.utils.weather import WeatherClient

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

EMPTY_INPUT_MESSAGE = "请至少提供基本档案或一项体检指标"
GENDER_MISSING_MESSAGE = "请补充性别信息以完成该项判定"
GENDER_INVALID_MESSAGE = "性别取值不合法"
HPYLORI_INVALID_MESSAGE = "检测结果取值不合法"
GENERIC_INVALID_MESSAGE = "请求参数不合法，请检查输入格式"
MEAL_MISSING_MESSAGE = "请提供用餐餐次"
MEAL_INVALID_MESSAGE = "餐次取值不合法"
SCENE_MISSING_MESSAGE = "请提供运动场景"
SCENE_INVALID_MESSAGE = "场景取值不合法"
MINUTES_MISSING_MESSAGE = "请提供可用运动时长"
MINUTES_INVALID_MESSAGE = "可用运动时长数值异常"

_PROFILE_RANGE_MESSAGES = {
    "profile.age": "年龄数值异常，请核对",
    "profile.height": "身高数值异常，请核对",
    "profile.weight": "体重数值异常，请核对",
}


def _body_has_any_data(body: Any) -> bool:
    """判断请求体中除性别外是否存在任何有效档案或体检指标数据。"""
    if not isinstance(body, dict):
        return bool(body)
    profile = body.get("profile")
    metrics = body.get("metrics")
    if isinstance(profile, dict) and any(
        profile.get(key) is not None for key in ("age", "height", "weight")
    ):
        return True
    if isinstance(metrics, dict) and any(value is not None for value in metrics.values()):
        return True
    return False


def _translate_validation_error(error: dict, body: Any) -> str:
    """将 Pydantic 校验错误映射为面向用户的中文提示（spec 5.2 / 5.3.3.4 / 5.4.3.5）。"""
    loc = tuple(part for part in error.get("loc", ()) if isinstance(part, str) and part != "body")
    field_path = ".".join(loc)
    error_type = error.get("type", "")

    if field_path in ("profile", "metrics") and error_type == "missing":
        return EMPTY_INPUT_MESSAGE

    if field_path == "profile.gender":
        if error_type == "missing":
            if _body_has_any_data(body):
                return GENDER_MISSING_MESSAGE
            return EMPTY_INPUT_MESSAGE
        return GENDER_INVALID_MESSAGE

    if field_path in _PROFILE_RANGE_MESSAGES and error_type != "missing":
        return _PROFILE_RANGE_MESSAGES[field_path]

    if field_path == "metrics.h_pylori":
        return HPYLORI_INVALID_MESSAGE

    if field_path == "meal":
        return MEAL_MISSING_MESSAGE if error_type == "missing" else MEAL_INVALID_MESSAGE

    if field_path == "scene":
        return SCENE_MISSING_MESSAGE if error_type == "missing" else SCENE_INVALID_MESSAGE

    if field_path == "available_minutes":
        return MINUTES_MISSING_MESSAGE if error_type == "missing" else MINUTES_INVALID_MESSAGE

    if field_path == "checked_sc_ids":
        if error_type in ("missing", "too_short", "empty"):
            return "请至少勾选一条自查条目"
        return "存在未识别的自查条目，已被忽略"

    if field_path == "duration":
        return "持续时长取值不合法"

    if field_path == "behavior_events":
        if error_type == "missing":
            return "请提供对话文本或行为事件"
        return "存在未识别的行为事件，已被忽略"

    if field_path == "text" and error_type == "string_type":
        return "请提供对话文本或行为事件"

    return GENERIC_INVALID_MESSAGE


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """档案级校验失败 → HTTP 422，携带中文提示（spec 5.2.1 / 5.2.3）。

    chat 路径分支（决策 D9）：化解 text 字段与 mood 端点的映射冲突——
    /api/v1/chat 请求走专用映射，既有 7 端点路径永不命中该分支。
    """
    body = getattr(exc, "body", None)
    seen: dict[str, dict] = {}
    for error in exc.errors():
        field = ".".join(str(part) for part in error.get("loc", ()) if part != "body")
        if request.url.path.startswith("/api/v1/chat"):
            message = _translate_chat_validation_error(error)
        else:
            message = _translate_validation_error(error, body)
        seen[f"{field}|{message}"] = {"field": field, "msg": message}
    return JSONResponse(status_code=422, content={"detail": list(seen.values())})


def _translate_chat_validation_error(error: dict) -> str:
    """chat 端点专用校验错误中文提示映射（spec 5.7.3.1）。"""
    loc = tuple(part for part in error.get("loc", ()) if isinstance(part, str) and part != "body")
    field_path = ".".join(loc)
    error_type = error.get("type", "")

    if field_path == "user_id":
        if error_type in ("missing", "too_short", "string_too_short", "empty", "string_type"):
            return "请提供用户标识"
        return GENERIC_INVALID_MESSAGE

    if field_path == "text":
        if error_type in ("missing", "too_short", "string_too_short", "empty"):
            return "请输入对话内容"
        return GENERIC_INVALID_MESSAGE

    return GENERIC_INVALID_MESSAGE


def _configure_audit_logging() -> None:
    """审计日志输出至本地日志文件，不记录指标具体数值（design.md 决策 D7）。"""
    audit_logger = logging.getLogger("coremate.audit")
    if any(isinstance(handler, logging.FileHandler) for handler in audit_logger.handlers):
        return
    log_dir = _PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    handler = logging.FileHandler(log_dir / "audit.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s"))
    audit_logger.addHandler(handler)
    audit_logger.setLevel(logging.INFO)


def create_app() -> FastAPI:
    """FastAPI 应用工厂：注册路由、引擎单例、静态目录与异常处理器。"""
    application = FastAPI(
        title="芯动管家 CoreMate",
        description="校园健康管理 Agent · 医学规则引擎、推荐、心理干预、Agent 对话编排与真实外部服务（含 Web 演示界面）",
        version="1.7.0",
    )
    application.state.engine = create_default_engine()
    application.state.diet_engine = create_default_diet_engine()
    application.state.exercise_engine = create_default_exercise_engine()
    application.state.course_advisor = create_default_sports_course_advisor()
    application.state.selfcheck_engine = create_default_selfcheck_engine()
    application.state.signal_engine = create_default_signal_engine()
    application.state.crisis_responder = create_default_crisis_responder()
    application.state.weather_client = WeatherClient()
    application.state.llm_client = LLMClient()
    application.state.vision_client = VisionClient()
    application.state.food_verdict_service = create_default_food_verdict_service()
    application.state.agent_router = create_default_agent_router(
        signal_engine=application.state.signal_engine,
        crisis_responder=application.state.crisis_responder,
        diet_engine=application.state.diet_engine,
        exercise_engine=application.state.exercise_engine,
        weather_client=application.state.weather_client,
        llm_client=application.state.llm_client,
        food_verdict_service=application.state.food_verdict_service,
    )
    application.include_router(health_router)
    application.include_router(recommend_router)
    application.include_router(mood_router)
    application.include_router(chat_router)
    application.include_router(pages_router)

    static_dir = _PROJECT_ROOT / "static"
    if static_dir.is_dir():
        application.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    application.add_exception_handler(RequestValidationError, validation_exception_handler)
    _configure_audit_logging()
    return application


app = create_app()