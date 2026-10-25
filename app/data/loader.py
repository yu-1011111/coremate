"""数据加载器：JSON 文件加载与 Markdown 表格文本解析。

load_records 解析内嵌的 Markdown 管道表格文本（raw_tables.py 常量），
运行期不依赖工作区外部文件路径（spec 4.5.3）。
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("coremate.data")


class DataLoadError(Exception):
    """数据加载失败（文件缺失、格式非法等）。"""


def load_json(path: str | Path) -> Any:
    """加载本地 JSON 数据文件，返回解析后的结构化数据。"""
    file_path = Path(path)
    if not file_path.is_file():
        raise DataLoadError(f"数据文件不存在: {file_path}")
    try:
        with file_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise DataLoadError(f"JSON 数据文件解析失败: {file_path}") from exc


_ROW_SPLIT_PATTERN = re.compile(r"(?<!\\)\|")


def _split_row_cells(line: str) -> list[Optional[str]]:
    """按未转义竖线拆分单元格；`\\|`（字面两字符）保留在值内由上层拆分。"""
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    cells = [cell.strip() for cell in _ROW_SPLIT_PATTERN.split(stripped)]
    return [cell if cell else None for cell in cells]


def _is_separator_row(line: str) -> bool:
    """识别 Markdown 表格的 `| --- | --- |` 分隔行。"""
    content = line.strip().replace("|", "").strip()
    if not content:
        return False
    return set(content) <= {"-", " ", ":"}


def load_records(table_text: str) -> list[dict[str, Optional[str]]]:
    """解析 Markdown 管道表格文本为记录列表（字段名 → 字符串值或 None）。

    规则：提取首个以 `|` 起始的表头行 → 跳过分隔行 → 数据行按未转义
    竖线拆分并去除首尾空白 → 空单元格归一为 None；数据行单元格数与表头
    列数不一致时跳过该行并记录告警（不阻断整体解析，spec 4.2.4）；
    文本无表头时抛 ValueError。
    """
    lines = table_text.split("\n")
    header_idx: Optional[int] = None
    for idx, line in enumerate(lines):
        if line.strip().startswith("|"):
            header_idx = idx
            break
    if header_idx is None:
        raise ValueError("表格文本中未找到以 | 起始的表头行")

    header_cells = _split_row_cells(lines[header_idx])
    records: list[dict[str, Optional[str]]] = []
    for line in lines[header_idx + 1 :]:
        if not line.strip().startswith("|"):
            continue
        if _is_separator_row(line):
            continue
        cells = _split_row_cells(line)
        if len(cells) != len(header_cells):
            logger.warning(
                "数据行单元格数 %d 与表头列数 %d 不一致，跳过该行",
                len(cells),
                len(header_cells),
            )
            continue
        records.append({name: cells[pos] for pos, name in enumerate(header_cells)})
    return records