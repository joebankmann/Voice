from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from voice.voices import VoiceInfo

try:
    import tkinter as tk
    from tkinter import scrolledtext, ttk
except ModuleNotFoundError:
    tk = None
    scrolledtext = None
    ttk = None


@dataclass
class UiController:
    pipeline: Any
    on_start_listening: Callable[[], None] | None = None
    on_stop_listening: Callable[[], None] | None = None
    on_voice_selected: Callable[[VoiceInfo], None] | None = None
    voices: list[VoiceInfo] = field(default_factory=list)
    selected_voice: str = ""
    status: str = "IDLE"
    transcript_lines: list[str] = field(default_factory=list)
    rendered_line_count: int = 0

    def __post_init__(self) -> None:
        self.pipeline.add_listener(self.handle_event)
        if not self.selected_voice and self.voices:
            self.selected_voice = self.voices[0].name

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

    def on_voice_change(self, voice_name: str) -> None:
        self.selected_voice = voice_name
        if self.on_voice_selected is None:
            return
        for voice in self.voices:
            if voice.name == voice_name:
                self.on_voice_selected(voice)
                return

    def pending_transcript_lines(self) -> list[str]:
        """Return only lines not yet shown in the transcript widget."""
        if self.rendered_line_count >= len(self.transcript_lines):
            return []
        pending = self.transcript_lines[self.rendered_line_count :]
        self.rendered_line_count = len(self.transcript_lines)
        return pending

    def handle_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type == "state":
            self.status = str(event.get("state", self.status))
        elif event_type == "error":
            self.status = str(event.get("text", "Pipeline error"))
        elif event_type == "user_transcript":
            self.transcript_lines.append(f"You: {event['text']}")
        elif event_type == "assistant_final":
            self.transcript_lines.append(f"Assistant: {event['text']}")


def run_app(
    pipeline: Any,
    *,
    on_start_listening: Callable[[], None] | None = None,
    on_stop_listening: Callable[[], None] | None = None,
    voices: list[VoiceInfo] | None = None,
    selected_voice: str = "",
    on_voice_selected: Callable[[VoiceInfo], None] | None = None,
) -> None:
    if tk is None or scrolledtext is None or ttk is None:
        raise RuntimeError("Tkinter is required to run the desktop UI")

    voice_list = list(voices or [])
    controller = UiController(
        pipeline=pipeline,
        on_start_listening=on_start_listening,
        on_stop_listening=on_stop_listening,
        on_voice_selected=on_voice_selected,
        voices=voice_list,
        selected_voice=selected_voice,
    )
    root = tk.Tk()
    root.title("Voice")
    root.geometry("520x680")

    status_var = tk.StringVar(value=controller.status)
    tk.Label(root, textvariable=status_var, font=("Helvetica", 16)).pack(pady=8)

    voice_row = tk.Frame(root)
    voice_row.pack(fill=tk.X, padx=12, pady=4)
    tk.Label(voice_row, text="Voice").pack(side=tk.LEFT)
    voice_names = [voice.name for voice in voice_list] or ["(no .onnx voices found)"]
    voice_var = tk.StringVar(
        value=controller.selected_voice if controller.selected_voice in voice_names else voice_names[0]
    )
    voice_menu = ttk.Combobox(
        voice_row,
        textvariable=voice_var,
        values=voice_names,
        state="readonly" if voice_list else "disabled",
        width=36,
    )
    voice_menu.pack(side=tk.LEFT, padx=8)

    def on_voice_picked(_event: object | None = None) -> None:
        controller.on_voice_change(voice_var.get())

    voice_menu.bind("<<ComboboxSelected>>", on_voice_picked)

    # state=normal keeps text selectable/copyable; we append instead of rewriting.
    transcript = scrolledtext.ScrolledText(
        root,
        wrap=tk.WORD,
        height=26,
        state=tk.NORMAL,
        exportselection=True,
    )
    transcript.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)
    last_status = {"value": controller.status}

    def refresh() -> None:
        if controller.status != last_status["value"]:
            last_status["value"] = controller.status
            status_var.set(controller.status)

        pending = controller.pending_transcript_lines()
        if pending:
            # Preserve user selection/scroll: only append new lines.
            at_bottom = transcript.yview()[1] >= 0.99
            prefix = "" if transcript.index("end-1c") == "1.0" else "\n"
            transcript.insert(tk.END, prefix + "\n".join(pending))
            if at_bottom:
                transcript.see(tk.END)
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
