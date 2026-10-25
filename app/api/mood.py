"""心理干预 REST API：便签自查与情绪信号识别端点（spec 5.2/5.3）。

M4 分流在 API 层编排（决策 D8）：SignalResult.mood_level 为 M4 时
调用 CrisisResponder.respond 并填入 SignalResult.crisis 字段，
同一响应内返回完整危机响应；M4 时 mood_resources 恒为空。
"""

from fastapi import APIRouter, Depends, HTTPException, Request

from app.rules.crisis import CrisisResponder
from app.rules.mood_schemas import (
    SelfcheckInvalidInputError,
    SelfcheckRequest,
    SelfcheckResult,
    SignalEmptyInputError,
    SignalRequest,
    SignalResult,
)
from app.rules.selfcheck import SelfcheckEngine
from app.rules.signal_engine import SignalEngine

router = APIRouter(prefix="/api/v1/mood", tags=["mood"])


def get_selfcheck_engine(request: Request) -> SelfcheckEngine:
    """FastAPI 依赖：从应用状态提取便签自查引擎单例（启动期装配）。"""
    return request.app.state.selfcheck_engine


def get_signal_engine(request: Request) -> SignalEngine:
    """FastAPI 依赖：从应用状态提取情绪信号识别引擎单例。"""
    return request.app.state.signal_engine


def get_crisis_responder(request: Request) -> CrisisResponder:
    """FastAPI 依赖：从应用状态提取危机响应器单例。"""
    return request.app.state.crisis_responder


@router.post("/selfcheck", response_model=SelfcheckResult)
def evaluate_selfcheck(
    payload: SelfcheckRequest,
    engine: SelfcheckEngine = Depends(get_selfcheck_engine),
) -> SelfcheckResult:
    """便签自查：勾选 SC1-SC9 + 持续时长 → 档位 + 手册应对 + 转介入口。"""
    try:
        return engine.evaluate(payload.checked_sc_ids, payload.duration)
    except SelfcheckInvalidInputError:
        raise HTTPException(status_code=400, detail="未识别任何有效自查条目") from None


@router.post("/signal", response_model=SignalResult)
def analyze_signal(
    payload: SignalRequest,
    engine: SignalEngine = Depends(get_signal_engine),
    responder: CrisisResponder = Depends(get_crisis_responder),
) -> SignalResult:
    """情绪信号识别：文本/行为/自报 → 档位判定；M4 时附完整危机响应。"""
    try:
        result = engine.analyze(payload.text, payload.behavior_events, payload.self_reports)
    except SignalEmptyInputError:
        raise HTTPException(status_code=422, detail="请提供对话文本或行为事件") from None

    if result.mood_level == "M4":
        result = result.model_copy(update={"crisis": responder.respond(result.matched_signals)})
    return result