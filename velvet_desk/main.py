from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .ui import VelvetDeskApp


def main() -> None:
    root = tk.Tk()
    # ttk theme
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass

    VelvetDeskApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
