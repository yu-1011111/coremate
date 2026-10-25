"""推荐 REST API：饮食/运动/选课建议三个端点（spec 5.3-5.5）。

suppression=True 时 diet/exercise 端点分流至引擎抑制构造返回空推荐
（M4 危机响应，design 决策 D4 L3 层）；缺省 False 时原有行为不变。
"""

import logging

from fastapi import APIRouter, Depends, Request

from app.rules.diet import DietRecommendEngine
from app.rules.exercise import ExerciseRecommendEngine
from app.rules.rec_schemas import (
    DietRecommendRequest,
    DietRecommendResult,
    ExerciseRecommendRequest,
    ExerciseRecommendResult,
    SportsCourseRecommendRequest,
    SportsCourseRecommendResult,
)
from app.rules.sports_course import SportsCourseAdvisor

router = APIRouter(prefix="/api/v1/recommend", tags=["recommend"])

audit_logger = logging.getLogger("coremate.audit")


def get_diet_engine(request: Request) -> DietRecommendEngine:
    """FastAPI 依赖：从应用状态提取饮食推荐引擎单例（启动期装配）。"""
    return request.app.state.diet_engine


def get_exercise_engine(request: Request) -> ExerciseRecommendEngine:
    """FastAPI 依赖：从应用状态提取运动推荐引擎单例。"""
    return request.app.state.exercise_engine


def get_course_advisor(request: Request) -> SportsCourseAdvisor:
    """FastAPI 依赖：从应用状态提取选课建议引擎单例。"""
    return request.app.state.course_advisor


@router.post("/diet", response_model=DietRecommendResult)
def recommend_diet(
    payload: DietRecommendRequest,
    engine: DietRecommendEngine = Depends(get_diet_engine),
) -> DietRecommendResult:
    """饮食推荐：健康标签 + 餐次 → 推荐/避雷菜品与饮品判定结果。"""
    if payload.suppression:
        audit_logger.info("饮食推荐被危机响应抑制")
        return engine.recommend_suppressed()
    return engine.recommend(payload.tags, payload.meal)


@router.post("/exercise", response_model=ExerciseRecommendResult)
def recommend_exercise(
    payload: ExerciseRecommendRequest,
    engine: ExerciseRecommendEngine = Depends(get_exercise_engine),
) -> ExerciseRecommendResult:
    """运动推荐：健康标签 + 场景 + 时长 + 可选天气 → 推荐动作与课表降级方案。"""
    if payload.suppression:
        audit_logger.info("运动推荐被危机响应抑制")
        return engine.recommend_suppressed()
    return engine.recommend(payload.tags, payload.scene, payload.available_minutes, payload.weather)


@router.post("/sports-course", response_model=SportsCourseRecommendResult)
def recommend_sports_course(
    payload: SportsCourseRecommendRequest,
    advisor: SportsCourseAdvisor = Depends(get_course_advisor),
) -> SportsCourseRecommendResult:
    """体育课选课建议：健康标签 → 推荐/避雷课目与谨慎提示。"""
    return advisor.recommend(payload.tags)