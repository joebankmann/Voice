from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

try:
    import tkinter as tk
    from tkinter import scrolledtext
except ModuleNotFoundError:
    tk = None
    scrolledtext = None


@dataclass
class UiController:
    pipeline: Any
    on_start_listening: Callable[[], None] | None = None
    on_stop_listening: Callable[[], None] | None = None
    status: str = "IDLE"
    transcript_lines: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.pipeline.add_listener(self.handle_event)

    def on_start(self) -> None:
        if self.on_start_listening is not None:
            self.on_start_listening()
            return
        self.pipeline.start()

    def on_stop(self) -> None:
        if self.on_stop_listening is not None:
            self.on_stop_listening()
            return
        self.pipeline.stop()

    def on_interrupt(self) -> None:
        self.pipeline.interrupt()

    def handle_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type == "state":
            self.status = str(event.get("state", self.status))
        elif event_type == "user_transcript":
            self.transcript_lines.append(f"You: {event['text']}")
        elif event_type == "assistant_final":
            self.transcript_lines.append(f"Assistant: {event['text']}")


def run_app(
    pipeline: Any,
    *,
    on_start_listening: Callable[[], None] | None = None,
    on_stop_listening: Callable[[], None] | None = None,
) -> None:
    if tk is None or scrolledtext is None:
        raise RuntimeError("Tkinter is required to run the desktop UI")

    controller = UiController(
        pipeline=pipeline,
        on_start_listening=on_start_listening,
        on_stop_listening=on_stop_listening,
    )
    root = tk.Tk()
    root.title("Voice")
    root.geometry("520x640")

    status_var = tk.StringVar(value=controller.status)
    tk.Label(root, textvariable=status_var, font=("Helvetica", 16)).pack(pady=8)

    transcript = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=28)
    transcript.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

    def refresh() -> None:
        status_var.set(controller.status)
        transcript.delete("1.0", tk.END)
        transcript.insert(tk.END, "\n".join(controller.transcript_lines))
        root.after(100, refresh)

    button_row = tk.Frame(root)
    button_row.pack(pady=8)
    tk.Button(button_row, text="Start", command=controller.on_start).pack(
        side=tk.LEFT,
        padx=4,
    )
    tk.Button(button_row, text="Stop", command=controller.on_stop).pack(
        side=tk.LEFT,
        padx=4,
    )
    tk.Button(button_row, text="Interrupt", command=controller.on_interrupt).pack(
        side=tk.LEFT,
        padx=4,
    )

    def close_window() -> None:
        controller.on_stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close_window)
    refresh()
    root.mainloop()
