from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


FieldType = Literal["file", "dir", "text", "bool"]


@dataclass(frozen=True)
class ModuleField:
    key: str
    label: str
    type: FieldType
    required: bool = True
    default: Optional[Any] = None
    help: str = ""


@dataclass(frozen=True)
class ModuleSpec:
    id: str
    name: str
    description: str
    # command is a list, e.g. ["python", "tools/tax_pipeline.py", "--in", "{input_dir}", "--out", "{output_dir}"]
    command: List[str]
    fields: List[ModuleField] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
