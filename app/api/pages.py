"""页面路由：GET / 渲染 Web 演示界面（frontend_ui spec 5.1）。

Jinja2Templates 模块级单例（设计决策 D7）；免责声明从第一阶段
schemas 只读导入（单一事实源，决策 D8）；端点函数体内仅含模板渲染，
禁止内联任何业务判定逻辑（spec 5.1.1.7）。
"""

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.rules.schemas import DISCLAIMER_TEXT

PAGE_TITLE = "芯动管家"

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

router = APIRouter(tags=["pages"])


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
def index(request: Request) -> HTMLResponse:
    """渲染主界面：标题与免责声明经上下文注入（无业务判定）。"""
    return templates.TemplateResponse(
        request,
        "index.html",
        {"title": PAGE_TITLE, "disclaimer": DISCLAIMER_TEXT},
    )