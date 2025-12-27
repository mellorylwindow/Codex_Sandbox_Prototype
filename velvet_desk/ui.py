"""
Velvet Desk UI (complete rewrite)
- Task + Style dropdowns
- "Type like a human" request box
- Log tab (Soulseek-ish)
- Output browser (out/ tree)
- Library tab with Rebuild Index
- Optional Everything-like search (uses es.exe if present; else Python scan)

No patches: replace this file entirely.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from tkinter import BOTH, END, LEFT, RIGHT, VERTICAL, X, Y, filedialog, messagebox
import tkinter as tk
from tkinter import ttk


# ----------------------------
# Helpers
# ----------------------------

def _now_stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def _safe_read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _open_path_in_explorer(path: Path) -> None:
    try:
        if path.exists():
            os.startfile(str(path))  # Windows
        else:
            messagebox.showwarning("Missing path", f"Path does not exist:\n{path}")
    except Exception as e:
        messagebox.showerror("Open failed", str(e))

def _repo_root_from_here() -> Path:
    """
    Walk upward until we find a .git folder or pyproject.toml.
    Falls back to cwd.
    """
    here = Path.cwd().resolve()
    for p in [here] + list(here.parents):
        if (p / ".git").exists() or (p / "pyproject.toml").exists():
            return p
    return here

def _rel_or_abs(repo_root: Path, maybe_rel: str) -> Path:
    p = Path(maybe_rel)
    return (repo_root / p).resolve() if not p.is_absolute() else p.resolve()

def _short(p: Path, repo_root: Path) -> str:
    try:
        return str(p.relative_to(repo_root))
    except Exception:
        return str(p)

def _which(cmd: str) -> str | None:
    from shutil import which
    return which(cmd)


# ----------------------------
# Data models
# ----------------------------

@dataclass
class StylePreset:
    id: str
    name: str
    path: Path

@dataclass
class TaskPreset:
    slug: str
    name: str
    path: Path


# ----------------------------
# Main App
# ----------------------------

class VelvetDeskApp:
    """
    main.py should do:
        root = tk.Tk()
        VelvetDeskApp(root)
        root.mainloop()
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.repo_root = _repo_root_from_here()

        self.root.title("Velvet Desk")
        self.root.geometry("1100x720")

        self._log_queue: queue.Queue[str] = queue.Queue()
        self._proc_thread: threading.Thread | None = None
        self._running = False

        # Paths (defaults; user can override in Settings tab later)
        self.paths = {
            "repo_root": str(self.repo_root),
            # TikTok inputs:
            "tiktok_images_ingested": str(self.repo_root / "notes" / "tiktok" / "images_ingested"),
            "tiktok_library_root": str(self.repo_root / "notes" / "tiktok" / "library"),
            "tiktok_library_index": str(self.repo_root / "notes" / "tiktok" / "library_index.json"),
            "tiktok_tasks_dir": str(self.repo_root / "notes" / "tiktok" / "tasks"),
            "tiktok_styles_dir": str(self.repo_root / "notes" / "tiktok" / "styles"),
            # TikTok outputs:
            "tiktok_out_root": str(self.repo_root / "out" / "tiktok_prompts"),
            # Prompt library outputs (generated today):
            "prompt_library_out": str(self.repo_root / "out" / "prompt_library"),
            # Generic out:
            "out_root": str(self.repo_root / "out"),
        }

        self._build_ui()
        self._tick_log_pump()
        self._refresh_all_lists()
        self._refresh_out_tree()

        self._log(f"🧠 Repo root: {self.repo_root}")
        self._log("Ready.")

    # ----------------------------
    # UI Build
    # ----------------------------

    def _build_ui(self) -> None:
        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill=BOTH, expand=True)

        self.tab_run = ttk.Frame(self.nb)
        self.tab_library = ttk.Frame(self.nb)
        self.tab_search = ttk.Frame(self.nb)
        self.tab_outputs = ttk.Frame(self.nb)
        self.tab_settings = ttk.Frame(self.nb)
        self.tab_log = ttk.Frame(self.nb)

        self.nb.add(self.tab_run, text="Run")
        self.nb.add(self.tab_library, text="Library")
        self.nb.add(self.tab_search, text="Search")
        self.nb.add(self.tab_outputs, text="Outputs")
        self.nb.add(self.tab_settings, text="Settings")
        self.nb.add(self.tab_log, text="Log")

        self._build_run_tab(self.tab_run)
        self._build_library_tab(self.tab_library)
        self._build_search_tab(self.tab_search)
        self._build_outputs_tab(self.tab_outputs)
        self._build_settings_tab(self.tab_settings)
        self._build_log_tab(self.tab_log)

    # ----------------------------
    # Log
    # ----------------------------

    def _build_log_tab(self, parent: ttk.Frame) -> None:
        top = ttk.Frame(parent)
        top.pack(fill=X, padx=10, pady=10)

        ttk.Button(top, text="Clear Log", command=self._clear_log).pack(side=LEFT)
        ttk.Button(top, text="Open out/", command=lambda: _open_path_in_explorer(_rel_or_abs(self.repo_root, self.paths["out_root"]))).pack(side=LEFT, padx=8)

        self.log_text = tk.Text(parent, wrap="word", height=30)
        self.log_text.pack(fill=BOTH, expand=True, padx=10, pady=(0,10))

    def _clear_log(self) -> None:
        self.log_text.delete("1.0", END)
        self._log("🧹 Log cleared")

    def _log(self, msg: str) -> None:
        self._log_queue.put(f"[{_now_stamp()}] {msg}")

    def _tick_log_pump(self) -> None:
        try:
            while True:
                line = self._log_queue.get_nowait()
                self.log_text.insert(END, line + "\n")
                self.log_text.see(END)
        except queue.Empty:
            pass
        self.root.after(120, self._tick_log_pump)

    # ----------------------------
    # Run Tab
    # ----------------------------

    def _build_run_tab(self, parent: ttk.Frame) -> None:
        container = ttk.Frame(parent)
        container.pack(fill=BOTH, expand=True, padx=10, pady=10)

        # Top controls row
        row = ttk.Frame(container)
        row.pack(fill=X)

        ttk.Label(row, text="Project:").pack(side=LEFT)
        self.project_var = tk.StringVar(value="TikTok — Prompt Pipeline")
        self.project_combo = ttk.Combobox(
            row,
            textvariable=self.project_var,
            values=[
                "TikTok — Prompt Pipeline",
                "Transcribe (manual CLI)",
                "Open Prompt Library Output",
            ],
            state="readonly",
            width=28,
        )
        self.project_combo.pack(side=LEFT, padx=8)

        ttk.Label(row, text="Task:").pack(side=LEFT, padx=(12, 0))
        self.task_var = tk.StringVar(value="")
        self.task_combo = ttk.Combobox(row, textvariable=self.task_var, values=[], state="readonly", width=34)
        self.task_combo.pack(side=LEFT, padx=8)

        ttk.Label(row, text="Style:").pack(side=LEFT, padx=(12, 0))
        self.style_var = tk.StringVar(value="")
        self.style_combo = ttk.Combobox(row, textvariable=self.style_var, values=[], state="readonly", width=28)
        self.style_combo.pack(side=LEFT, padx=8)

        ttk.Button(row, text="Refresh", command=self._refresh_all_lists).pack(side=RIGHT)

        # Human request box
        req_frame = ttk.LabelFrame(container, text="Type like a human")
        req_frame.pack(fill=BOTH, expand=False, pady=(12, 8))

        self.request_text = tk.Text(req_frame, height=6, wrap="word")
        self.request_text.pack(fill=BOTH, expand=True, padx=8, pady=8)
        self.request_text.insert(END, "Explain Velvet OS like a hobby to coworkers. Simple, non-weird. Benefits for neurodivergent brains.")

        # Buttons row
        btns = ttk.Frame(container)
        btns.pack(fill=X, pady=(6, 8))

        ttk.Button(btns, text="Open Images (ingested)", command=self._open_images_ingested).pack(side=LEFT)
        ttk.Button(btns, text="Import Images…", command=self._import_images).pack(side=LEFT, padx=8)
        ttk.Button(btns, text="Open TikTok Output", command=self._open_tiktok_out).pack(side=LEFT, padx=8)
        ttk.Button(btns, text="Open Prompt Library Output", command=self._open_prompt_library_out).pack(side=LEFT, padx=8)

        self.progress = ttk.Progressbar(btns, mode="indeterminate")
        self.progress.pack(side=RIGHT, fill=X, expand=True, padx=(12, 0))

        run_btns = ttk.Frame(container)
        run_btns.pack(fill=X, pady=(6, 0))

        ttk.Button(run_btns, text="▶ Run Selected Task", command=self._run_selected_task).pack(side=LEFT)
        ttk.Button(run_btns, text="▶ Run From Request (uses Style + Library)", command=self._run_from_request).pack(side=LEFT, padx=8)
        ttk.Button(run_btns, text="⏹ Stop (best effort)", command=self._stop_run).pack(side=LEFT, padx=8)

        # Small help
        help_txt = (
            "How it works:\n"
            "- Run Selected Task: runs notes/tiktok/tasks/<task>.json\n"
            "- Run From Request: generates a temp task + passes --request and --style\n"
            "- Library-aware mode expects notes/tiktok/library_index.json (build it in Library tab)\n"
        )
        ttk.Label(container, text=help_txt, justify="left").pack(fill=X, pady=(10, 0))

    def _open_images_ingested(self) -> None:
        p = _rel_or_abs(self.repo_root, self.paths["tiktok_images_ingested"])
        _ensure_dir(p)
        _open_path_in_explorer(p)

    def _open_tiktok_out(self) -> None:
        p = _rel_or_abs(self.repo_root, self.paths["tiktok_out_root"])
        _ensure_dir(p)
        _open_path_in_explorer(p)

    def _open_prompt_library_out(self) -> None:
        p = _rel_or_abs(self.repo_root, self.paths["prompt_library_out"])
        _ensure_dir(p)
        _open_path_in_explorer(p)

    def _import_images(self) -> None:
        """
        Lets you select images; copies them into notes/tiktok/images_ingested/
        """
        dest = _rel_or_abs(self.repo_root, self.paths["tiktok_images_ingested"])
        _ensure_dir(dest)
        files = filedialog.askopenfilenames(
            title="Select images to import",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp"), ("All files", "*.*")]
        )
        if not files:
            return

        copied = 0
        for f in files:
            src = Path(f)
            if not src.exists():
                continue
            target = dest / src.name
            # avoid overwrite
            if target.exists():
                target = dest / f"{src.stem}__{int(time.time())}{src.suffix}"
            shutil.copy2(src, target)
            copied += 1

        self._log(f"📥 Imported {copied} images into: {_short(dest, self.repo_root)}")

    def _run_selected_task(self) -> None:
        if self._running:
            messagebox.showwarning("Running", "A job is already running.")
            return

        task_path = self._get_selected_task_path()
        if not task_path:
            messagebox.showwarning("No task", "Pick a task from the dropdown (notes/tiktok/tasks).")
            return

        self._run_pipeline(task_path=task_path, request_override=None, style_override=None)

    def _run_from_request(self) -> None:
        if self._running:
            messagebox.showwarning("Running", "A job is already running.")
            return

        req = self.request_text.get("1.0", END).strip()
        if not req:
            messagebox.showwarning("Empty request", "Type a request first.")
            return

        style = self.style_var.get().strip() or None

        # Create a temp task json in out/.velvet_desk/tmp_tasks/
        tmp_dir = self.repo_root / "out" / ".velvet_desk" / "tmp_tasks"
        _ensure_dir(tmp_dir)

        slug = f"adhoc_{int(time.time())}"
        task = {
            "schema": "velvet.task.v1",
            "name": f"Adhoc: {req[:50]}",
            "slug": slug,
            "request": req,
            "style_id": style or "tiktok_default",
            "inputs": {
                "images_dir": "notes/tiktok/images_ingested",
                "notes_dir": "notes/tiktok/inbox"
            },
            "outputs": {
                "root": "out/tiktok_prompts"
            },
            "controls": {
                "take": 12,
                "teleprompter_top": 3,
                "min_score": 0.25
            }
        }

        task_path = tmp_dir / f"{slug}.json"
        task_path.write_text(json.dumps(task, indent=2), encoding="utf-8")

        self._log(f"🧾 Created temp task: {_short(task_path, self.repo_root)}")
        self._run_pipeline(task_path=task_path, request_override=req, style_override=style)

    def _stop_run(self) -> None:
        # Best effort: we can't safely kill subprocess without tracking the Popen.
        # This UI uses subprocess.run in a thread, so stop is just a UX signal.
        if not self._running:
            return
        self._log("⏹ Stop requested (best effort). If the script is busy, it may finish anyway.")

    def _get_selected_task_path(self) -> Path | None:
        val = self.task_var.get().strip()
        if not val:
            return None

        tasks_dir = _rel_or_abs(self.repo_root, self.paths["tiktok_tasks_dir"])
        p = tasks_dir / f"{val}.json"
        return p if p.exists() else None

    def _run_pipeline(self, task_path: Path, request_override: str | None, style_override: str | None) -> None:
        """
        Runs tools/tiktok_autorun.py with a task JSON.
        NOTE: This expects tiktok_autorun.py to accept:
            --task <json>
            --request <text>   (optional)
            --style <id>       (optional)
            --library-index <path> (optional)
        """
        self._running = True
        self.progress.start(10)

        script = self.repo_root / "tools" / "tiktok_autorun.py"
        if not script.exists():
            self._running = False
            self.progress.stop()
            messagebox.showerror("Missing script", f"Not found:\n{script}")
            return

        cmd = [sys.executable, str(script), "--task", str(task_path)]
        if request_override:
            cmd += ["--request", request_override]
        if style_override:
            cmd += ["--style", style_override]

        lib_index = _rel_or_abs(self.repo_root, self.paths["tiktok_library_index"])
        cmd += ["--library-index", str(lib_index)]

        self._log("▶ Running: TikTok — Prompt Pipeline")
        self._log(f"   cwd: {self.repo_root}")
        self._log(f"   cmd: {' '.join(cmd)}")

        def runner():
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(self.repo_root),
                    capture_output=True,
                    text=True
                )
                if proc.stdout:
                    for line in proc.stdout.splitlines():
                        self._log(line)
                if proc.stderr:
                    for line in proc.stderr.splitlines():
                        self._log(line)

                if proc.returncode == 0:
                    self._log("✅ Done.")
                else:
                    self._log(f"❌ Exit code: {proc.returncode}")
            except Exception as e:
                self._log(f"💥 Run failed: {e}")
            finally:
                self._running = False
                self.progress.stop()
                self._refresh_out_tree()

        self._proc_thread = threading.Thread(target=runner, daemon=True)
        self._proc_thread.start()

    # ----------------------------
    # Library Tab
    # ----------------------------

    def _build_library_tab(self, parent: ttk.Frame) -> None:
        container = ttk.Frame(parent)
        container.pack(fill=BOTH, expand=True, padx=10, pady=10)

        top = ttk.Frame(container)
        top.pack(fill=X)

        ttk.Button(top, text="Rebuild Library Index", command=self._rebuild_library_index).pack(side=LEFT)
        ttk.Button(top, text="Open Library Folder", command=self._open_library_folder).pack(side=LEFT, padx=8)
        ttk.Button(top, text="Open Styles Folder", command=self._open_styles_folder).pack(side=LEFT, padx=8)
        ttk.Button(top, text="Open Tasks Folder", command=self._open_tasks_folder).pack(side=LEFT, padx=8)

        # Library list
        mid = ttk.LabelFrame(container, text="Library Items (from library_index.json)")
        mid.pack(fill=BOTH, expand=True, pady=(12, 0))

        self.library_list = ttk.Treeview(mid, columns=("tags", "path"), show="headings")
        self.library_list.heading("tags", text="Tags")
        self.library_list.heading("path", text="Path")
        self.library_list.column("tags", width=240, anchor="w")
        self.library_list.column("path", width=760, anchor="w")
        self.library_list.pack(fill=BOTH, expand=True, padx=8, pady=8)

        self.library_list.bind("<Double-1>", self._on_library_open)

    def _open_library_folder(self) -> None:
        p = _rel_or_abs(self.repo_root, self.paths["tiktok_library_root"])
        _ensure_dir(p)
        _open_path_in_explorer(p)

    def _open_styles_folder(self) -> None:
        p = _rel_or_abs(self.repo_root, self.paths["tiktok_styles_dir"])
        _ensure_dir(p)
        _open_path_in_explorer(p)

    def _open_tasks_folder(self) -> None:
        p = _rel_or_abs(self.repo_root, self.paths["tiktok_tasks_dir"])
        _ensure_dir(p)
        _open_path_in_explorer(p)

    def _on_library_open(self, _evt) -> None:
        sel = self.library_list.selection()
        if not sel:
            return
        item = self.library_list.item(sel[0])
        path_str = item["values"][1]
        p = _rel_or_abs(self.repo_root, path_str)
        _open_path_in_explorer(p.parent)

    def _rebuild_library_index(self) -> None:
        """
        Very simple indexer:
        - scans notes/tiktok/library/*.md
        - extracts frontmatter id/title/tags if present
        - writes notes/tiktok/library_index.json
        """
        root = _rel_or_abs(self.repo_root, self.paths["tiktok_library_root"])
        _ensure_dir(root)

        items = []
        for md in sorted(root.rglob("*.md")):
            txt = md.read_text(encoding="utf-8", errors="replace")
            front = _parse_frontmatter(txt)
            preview = _first_preview(txt)

            items.append({
                "path": _short(md, self.repo_root),
                "id": front.get("id") or md.stem,
                "title": front.get("title") or md.stem,
                "tags": front.get("tags") or [],
                "mtime": int(md.stat().st_mtime),
                "size": int(md.stat().st_size),
                "preview": preview[:200]
            })

        idx = {
            "schema": "velvet.library_index.v1",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "root": _short(root, self.repo_root),
            "files": items
        }

        out = _rel_or_abs(self.repo_root, self.paths["tiktok_library_index"])
        out.write_text(json.dumps(idx, indent=2), encoding="utf-8")
        self._log(f"📚 Library index written: {_short(out, self.repo_root)} ({len(items)} items)")
        self._refresh_library_list()

    # ----------------------------
    # Search Tab (Everything-ish)
    # ----------------------------

    def _build_search_tab(self, parent: ttk.Frame) -> None:
        container = ttk.Frame(parent)
        container.pack(fill=BOTH, expand=True, padx=10, pady=10)

        row = ttk.Frame(container)
        row.pack(fill=X)

        ttk.Label(row, text="Query:").pack(side=LEFT)
        self.search_var = tk.StringVar(value="recording_10")
        ttk.Entry(row, textvariable=self.search_var, width=60).pack(side=LEFT, padx=8)

        ttk.Label(row, text="Root:").pack(side=LEFT, padx=(12, 0))
        self.search_root_var = tk.StringVar(value=str(self.repo_root))
        ttk.Entry(row, textvariable=self.search_root_var, width=42).pack(side=LEFT, padx=8)

        ttk.Button(row, text="Search", command=self._run_search).pack(side=LEFT, padx=8)

        self.search_mode_var = tk.StringVar(value="Auto (Everything if available)")
        self.search_mode_combo = ttk.Combobox(
            row,
            textvariable=self.search_mode_var,
            values=["Auto (Everything if available)", "Everything CLI (es.exe)", "Python Scan"],
            state="readonly",
            width=26
        )
        self.search_mode_combo.pack(side=RIGHT)

        results_box = ttk.LabelFrame(container, text="Results (double-click to open folder)")
        results_box.pack(fill=BOTH, expand=True, pady=(12, 0))

        self.search_results = ttk.Treeview(results_box, columns=("path",), show="headings")
        self.search_results.heading("path", text="Path")
        self.search_results.column("path", width=1000, anchor="w")
        self.search_results.pack(fill=BOTH, expand=True, padx=8, pady=8)
        self.search_results.bind("<Double-1>", self._on_search_open)

    def _run_search(self) -> None:
        q = self.search_var.get().strip()
        root = Path(self.search_root_var.get().strip() or str(self.repo_root))
        if not q:
            return

        for x in self.search_results.get_children():
            self.search_results.delete(x)

        mode = self.search_mode_var.get()
        use_everything = (mode != "Python Scan") and (_which("es.exe") is not None)

        if mode == "Everything CLI (es.exe)" and not use_everything:
            self._log("⚠ es.exe not found in PATH. Install Everything + CLI or use Python Scan.")
            use_everything = False

        if use_everything:
            self._log(f"🔎 Searching (Everything CLI): {q}")
            results = self._search_everything(q, root)
        else:
            self._log(f"🔎 Searching (Python scan): {q}")
            results = self._search_python(q, root)

        for p in results[:500]:
            self.search_results.insert("", END, values=(p,))

        self._log(f"🔎 Found {len(results)} results (showing up to 500)")

    def _search_everything(self, query: str, root: Path) -> list[str]:
        """
        Uses Everything CLI if installed: es.exe
        We filter by root by post-filtering (Everything supports 'path:' too, but keep it simple).
        """
        try:
            cmd = ["es.exe", query]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
            out = []
            root_s = str(root).lower()
            for ln in lines:
                if ln.lower().startswith(root_s):
                    out.append(ln)
            return out
        except Exception as e:
            self._log(f"Everything search failed: {e}")
            return []

    def _search_python(self, query: str, root: Path) -> list[str]:
        root = root.resolve()
        out = []
        ql = query.lower()
        try:
            for p in root.rglob("*"):
                if p.is_file() and ql in p.name.lower():
                    out.append(str(p))
        except Exception as e:
            self._log(f"Python scan failed: {e}")
        return out

    def _on_search_open(self, _evt) -> None:
        sel = self.search_results.selection()
        if not sel:
            return
        path_str = self.search_results.item(sel[0])["values"][0]
        p = Path(path_str)
        if p.exists():
            _open_path_in_explorer(p.parent)

    # ----------------------------
    # Outputs Tab
    # ----------------------------

    def _build_outputs_tab(self, parent: ttk.Frame) -> None:
        container = ttk.Frame(parent)
        container.pack(fill=BOTH, expand=True, padx=10, pady=10)

        top = ttk.Frame(container)
        top.pack(fill=X)

        ttk.Button(top, text="Refresh", command=self._refresh_out_tree).pack(side=LEFT)
        ttk.Button(top, text="Open out/", command=lambda: _open_path_in_explorer(_rel_or_abs(self.repo_root, self.paths["out_root"]))).pack(side=LEFT, padx=8)

        tree_frame = ttk.Frame(container)
        tree_frame.pack(fill=BOTH, expand=True, pady=(12, 0))

        self.out_tree = ttk.Treeview(tree_frame)
        yscroll = ttk.Scrollbar(tree_frame, orient=VERTICAL, command=self.out_tree.yview)
        self.out_tree.configure(yscrollcommand=yscroll.set)

        self.out_tree.pack(side=LEFT, fill=BOTH, expand=True)
        yscroll.pack(side=RIGHT, fill=Y)

        self.out_tree.bind("<Double-1>", self._on_out_open)

    def _refresh_out_tree(self) -> None:
        for x in self.out_tree.get_children():
            self.out_tree.delete(x)

        root = _rel_or_abs(self.repo_root, self.paths["out_root"])
        _ensure_dir(root)

        def insert_dir(parent_id: str, d: Path, depth: int = 0, max_depth: int = 4):
            if depth > max_depth:
                return
            try:
                entries = sorted(d.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
            except Exception:
                return

            for p in entries:
                rel = _short(p, self.repo_root)
                node_id = self.out_tree.insert(parent_id, END, text=p.name, values=(rel,))
                if p.is_dir():
                    insert_dir(node_id, p, depth + 1, max_depth)

        insert_dir("", root)

    def _on_out_open(self, _evt) -> None:
        sel = self.out_tree.selection()
        if not sel:
            return
        node = sel[0]
        vals = self.out_tree.item(node, "values")
        if not vals:
            return
        rel = vals[0]
        p = _rel_or_abs(self.repo_root, rel)
        if p.exists():
            _open_path_in_explorer(p if p.is_dir() else p.parent)

    # ----------------------------
    # Settings Tab
    # ----------------------------

    def _build_settings_tab(self, parent: ttk.Frame) -> None:
        container = ttk.Frame(parent)
        container.pack(fill=BOTH, expand=True, padx=10, pady=10)

        ttk.Label(container, text="Folder mapping (edit paths if you want; defaults are auto-mapped).").pack(anchor="w")

        self.settings_vars = {}
        grid = ttk.Frame(container)
        grid.pack(fill=BOTH, expand=True, pady=(10, 0))

        keys = [
            "tiktok_images_ingested",
            "tiktok_library_root",
            "tiktok_library_index",
            "tiktok_tasks_dir",
            "tiktok_styles_dir",
            "tiktok_out_root",
            "prompt_library_out",
            "out_root",
        ]

        for i, k in enumerate(keys):
            ttk.Label(grid, text=k).grid(row=i, column=0, sticky="w", pady=4)
            v = tk.StringVar(value=self.paths[k])
            self.settings_vars[k] = v
            ttk.Entry(grid, textvariable=v, width=110).grid(row=i, column=1, sticky="we", padx=8, pady=4)

        grid.columnconfigure(1, weight=1)

        btns = ttk.Frame(container)
        btns.pack(fill=X, pady=(12, 0))
        ttk.Button(btns, text="Save Settings (session only)", command=self._apply_settings).pack(side=LEFT)

        note = (
            "Note: This UI currently keeps settings in memory for the session.\n"
            "If you want persistence, we can add velvet_desk/settings.json next."
        )
        ttk.Label(container, text=note, justify="left").pack(anchor="w", pady=(10, 0))

    def _apply_settings(self) -> None:
        for k, v in self.settings_vars.items():
            self.paths[k] = v.get().strip()
        self._log("⚙ Settings applied for this session.")
        self._refresh_all_lists()
        self._refresh_out_tree()
        self._refresh_library_list()

    # ----------------------------
    # Lists refresh
    # ----------------------------

    def _refresh_all_lists(self) -> None:
        self._refresh_tasks()
        self._refresh_styles()
        self._refresh_library_list()

    def _refresh_tasks(self) -> None:
        tasks_dir = _rel_or_abs(self.repo_root, self.paths["tiktok_tasks_dir"])
        _ensure_dir(tasks_dir)

        tasks = []
        for p in sorted(tasks_dir.glob("*.json")):
            data = _safe_read_json(p) or {}
            slug = data.get("slug") or p.stem
            name = data.get("name") or slug
            tasks.append(TaskPreset(slug=slug, name=name, path=p))

        # Combobox shows slug (simple). We can enhance later to show "slug — name".
        self.task_combo["values"] = [t.slug for t in tasks]
        if tasks and not self.task_var.get():
            self.task_var.set(tasks[0].slug)

        self._log(f"↻ Loaded {len(tasks)} tasks from { _short(tasks_dir, self.repo_root) }")

    def _refresh_styles(self) -> None:
        styles_dir = _rel_or_abs(self.repo_root, self.paths["tiktok_styles_dir"])
        _ensure_dir(styles_dir)

        styles = []
        for p in sorted(styles_dir.glob("*.json")):
            data = _safe_read_json(p) or {}
            sid = data.get("id") or p.stem
            name = data.get("name") or sid
            styles.append(StylePreset(id=sid, name=name, path=p))

        self.style_combo["values"] = [s.id for s in styles]
        if styles and not self.style_var.get():
            self.style_var.set(styles[0].id)

        self._log(f"↻ Loaded {len(styles)} styles from { _short(styles_dir, self.repo_root) }")

    def _refresh_library_list(self) -> None:
        for x in self.library_list.get_children():
            self.library_list.delete(x)

        idx_path = _rel_or_abs(self.repo_root, self.paths["tiktok_library_index"])
        data = _safe_read_json(idx_path)
        if not data:
            return

        files = data.get("files") or []
        for f in files:
            tags = ", ".join(f.get("tags") or [])
            path = f.get("path") or ""
            self.library_list.insert("", END, values=(tags, path))


# ----------------------------
# Frontmatter parsing helpers
# ----------------------------

def _parse_frontmatter(text: str) -> dict:
    """
    Minimal YAML-ish frontmatter parser (key: value, tags: [a,b]).
    Avoids external deps.
    """
    text = text.lstrip()
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    block = text[3:end].strip().splitlines()
    out: dict = {}
    for ln in block:
        ln = ln.strip()
        if not ln or ln.startswith("#") or ":" not in ln:
            continue
        k, v = ln.split(":", 1)
        k = k.strip()
        v = v.strip()
        if v.startswith("[") and v.endswith("]"):
            inner = v[1:-1].strip()
            if inner:
                parts = [p.strip().strip('"').strip("'") for p in inner.split(",")]
                out[k] = [p for p in parts if p]
            else:
                out[k] = []
        else:
            out[k] = v.strip('"').strip("'")
    return out

def _first_preview(text: str) -> str:
    """
    Returns first non-frontmatter content line(s) for preview.
    """
    text = text.lstrip()
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            text = text[end+4:]
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return " ".join(lines[:2]) if lines else ""


# Backwards-compatible alias if older main imports VelvetDeskUI
VelvetDeskUI = VelvetDeskApp
