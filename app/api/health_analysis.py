"""健康分析 REST API：POST /api/v1/health/analysis 与 GET /api/v1/health。"""

from fastapi import APIRouter, Depends, Request

from app.rules.medical import MedicalRulesEngine
from app.rules.schemas import AnalysisRequest, AnalysisResult

router = APIRouter(prefix="/api/v1", tags=["health"])


def get_engine(request: Request) -> MedicalRulesEngine:
    """FastAPI 依赖：从应用状态获取引擎单例（启动期装配）。"""
    return request.app.state.engine


@router.get("/health")
def health_check() -> dict:
    """服务健康检查，不触达规则引擎。"""
    return {"status": "ok", "service": "coremate-medical-rules-engine"}


@router.post("/health/analysis", response_model=AnalysisResult)
def analyze_health(
    payload: AnalysisRequest,
    engine: MedicalRulesEngine = Depends(get_engine),
) -> AnalysisResult:
    """健康指标分析：档案级校验通过后调用本地医学规则引擎。"""
    return engine.analyze(payload.profile, payload.metrics)