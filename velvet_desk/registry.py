from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .models import ModuleField, ModuleSpec


class RegistryError(RuntimeError):
    pass


def _here() -> Path:
    return Path(__file__).resolve().parent


def load_registry(path: Path | None = None) -> List[ModuleSpec]:
    reg_path = path or (_here() / "modules.json")
    if not reg_path.exists():
        raise RegistryError(f"Registry file not found: {reg_path}")

    data = json.loads(reg_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "modules" not in data:
        raise RegistryError("modules.json must be an object with a top-level 'modules' list")

    modules_raw = data["modules"]
    if not isinstance(modules_raw, list):
        raise RegistryError("'modules' must be a list")

    out: List[ModuleSpec] = []
    seen_ids = set()

    for m in modules_raw:
        if not isinstance(m, dict):
            raise RegistryError("Each module must be an object")

        mid = str(m.get("id", "")).strip()
        if not mid:
            raise RegistryError("Module missing 'id'")
        if mid in seen_ids:
            raise RegistryError(f"Duplicate module id: {mid}")
        seen_ids.add(mid)

        name = str(m.get("name", "")).strip()
        desc = str(m.get("description", "")).strip()
        cmd = m.get("command")

        if not name:
            raise RegistryError(f"Module {mid} missing 'name'")
        if not desc:
            desc = ""
        if not isinstance(cmd, list) or not all(isinstance(x, str) for x in cmd):
            raise RegistryError(f"Module {mid} 'command' must be a list of strings")

        fields_raw = m.get("fields", [])
        fields: List[ModuleField] = []
        if fields_raw:
            if not isinstance(fields_raw, list):
                raise RegistryError(f"Module {mid} 'fields' must be a list")
            for f in fields_raw:
                if not isinstance(f, dict):
                    raise RegistryError(f"Module {mid} has a field that is not an object")
                key = str(f.get("key", "")).strip()
                label = str(f.get("label", "")).strip()
                ftype = str(f.get("type", "")).strip()
                required = bool(f.get("required", True))
                default = f.get("default", None)
                help_txt = str(f.get("help", "")).strip()
                if not key or not label or not ftype:
                    raise RegistryError(f"Module {mid} has invalid field (key/label/type required)")
                if ftype not in {"file", "dir", "text", "bool"}:
                    raise RegistryError(f"Module {mid} field '{key}' has invalid type: {ftype}")
                fields.append(ModuleField(key=key, label=label, type=ftype, required=required, default=default, help=help_txt))

        tags = m.get("tags", [])
        if tags and (not isinstance(tags, list) or not all(isinstance(t, str) for t in tags)):
            raise RegistryError(f"Module {mid} 'tags' must be a list of strings")

        out.append(ModuleSpec(
            id=mid,
            name=name,
            description=desc,
            command=[c for c in cmd],
            fields=fields,
            tags=tags or [],
        ))

    return out
