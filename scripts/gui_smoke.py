"""Drive the real tkinter GUI of main.py with an auto-clicker (M0 acceptance aid).

Runs ``main.py`` unchanged (human vs human, default example decks) and clicks
real widgets: default decks -> confirm mulligans -> each turn try to play the
first playable card once, then end the turn -> close the final dialog.
Needs a display (on Linux: ``xvfb-run python scripts/gui_smoke.py``).
Exit code 0 means main.py ran to its normal end without an exception.
Options: --big (huge window), --seed N (seed the global RNG), --trace FILE (dump the clicked labels).
"""

import os
import runpy
import sys
import tkinter as tk
from tkinter import ttk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from src.common import text as T  # noqa: E402

BIG_WINDOW = "--big" in sys.argv
stats = {"clicks": 0, "prompts": {}, "tried_play": False}


def all_widgets(widget):
    yield widget
    for child in widget.winfo_children():
        yield from all_widgets(child)


def pick_button(buttons, prompt):
    labels = {b.cget("text"): b for b in buttons}
    for fixed in (T.DECK_START_DEFAULT, T.MULLIGAN_CONFIRM):
        if fixed in labels:
            return labels[fixed]
    if T.MENU_END_TURN in labels and T.MENU_PLAY_CARD in labels:  # main menu
        return labels[T.MENU_END_TURN] if stats["tried_play"] else labels[T.MENU_PLAY_CARD]
    return buttons[0]  # first card / OK / first effect option


def is_shown(widget):
    """True if the widget and all its ancestors are managed (packed/gridded)."""
    while widget is not None and not isinstance(widget, (tk.Tk, tk.Toplevel)):
        if not widget.winfo_manager():
            return False
        widget = widget.master
    return True


def poll(root):
    stats["polls"] = stats.get("polls", 0) + 1
    if BIG_WINDOW and id(root) not in stats.setdefault("resized", set()) and root.winfo_width() >= 1200:
        stats["resized"].add(id(root))
        # The stock 1200x800 window can clip the choice buttons off-screen when the
        # board gets tall (upstream layout issue). Give the harness a huge window.
        root.geometry("1900x2400")
    try:
        buttons = [w for w in all_widgets(root)
                   if isinstance(w, (tk.Button, ttk.Button)) and is_shown(w)]
        if buttons:
            prompt = ""
            for w in all_widgets(root):
                if isinstance(w, ttk.Label) and w.winfo_ismapped() and "---" in str(w.cget("text")):
                    prompt = w.cget("text")
            button = pick_button(buttons, prompt)
            if not button.winfo_ismapped():
                # Give Tk up to ~0.5 s to map it; after that treat it as clipped and click anyway.
                stats["unmapped_polls"] = stats.get("unmapped_polls", 0) + 1
                if stats["unmapped_polls"] < 100:
                    return
                stats["clipped"] = stats.get("clipped", 0) + 1
                stats.setdefault("clip_example", f"{button.cget('text')!r} window={root.winfo_width()}x{root.winfo_height()}")
            stats["unmapped_polls"] = 0
            if "(ID" in button.cget("text"):
                stats["card_clicks"] = stats.get("card_clicks", 0) + 1
            if button.cget("text") == T.MENU_PLAY_CARD:
                stats["tried_play"] = True
            elif button.cget("text") == T.MENU_END_TURN:
                stats["tried_play"] = False
            stats["clicks"] += 1
            stats.setdefault("trace", []).append(button.cget("text"))
            stats["prompts"][button.cget("text")[:20]] = stats["prompts"].get(button.cget("text")[:20], 0) + 1
            button.invoke()
    finally:
        try:
            root.after(5, poll, root)
        except tk.TclError:
            pass


_orig_init = tk.Tk.__init__


def _patched_init(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)
    self.after(50, poll, self)


tk.Tk.__init__ = _patched_init

if __name__ == "__main__":
    import io, contextlib, json, random
    if "--seed" in sys.argv:
        random.seed(int(sys.argv[sys.argv.index("--seed") + 1]))
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        runpy.run_path(os.path.join(ROOT, "main.py"), run_name="__main__")
    log = sink.getvalue()
    print(f"main.py finished normally. clicks={stats['clicks']}")
    print("card/target buttons clicked:", stats.get("card_clicks", 0))
    print("buttons that stayed unmapped (clipped) and were clicked anyway:", stats.get("clipped", 0), stats.get("clip_example", ""))
    print("turn starts:", sum(1 for l in log.splitlines() if "턴 시작 (시작 단계)" in l))
    if "--trace" in sys.argv:
        with open(sys.argv[sys.argv.index("--trace") + 1], "w", encoding="utf-8") as f:
            json.dump(stats["trace"], f, ensure_ascii=False)
    print("button labels clicked:", dict(sorted(stats["prompts"].items(), key=lambda kv: -kv[1])[:8]))
