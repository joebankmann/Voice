from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"[.!?…][\"')\]]*$")


class PhraseChunker:
    def __init__(self, max_chars: int = 120) -> None:
        self.max_chars = max_chars
        self._buf = ""

    def push(self, token: str) -> list[str]:
        self._buf += token
        return self._drain(force=False)

    def flush(self) -> list[str]:
        return self._drain(force=True)

    def _drain(self, force: bool) -> list[str]:
        emitted: list[str] = []
        while True:
            stripped = self._buf.strip()
            if not stripped:
                self._buf = ""
                break

            # Prefer sentence boundary
            for i, ch in enumerate(self._buf):
                if ch in ".!?…" and _SENTENCE_END.search(self._buf[: i + 1].rstrip()):
                    # include trailing quotes/brackets after punct if present
                    end = i + 1
                    while end < len(self._buf) and self._buf[end] in "\"')]\n ":
                        if self._buf[end] == "\n":
                            end += 1
                            break
                        end += 1
                    piece = self._buf[:end].strip()
                    self._buf = self._buf[end:]
                    if piece:
                        emitted.append(piece)
                    break
            else:
                if force and self._buf.strip():
                    emitted.append(self._buf.strip())
                    self._buf = ""
                elif len(self._buf) >= self.max_chars:
                    # soft break at last space
                    cut = self._buf.rfind(" ", 0, self.max_chars)
                    if cut <= 0:
                        cut = self.max_chars
                    piece = self._buf[:cut].strip()
                    self._buf = self._buf[cut:]
                    if piece:
                        emitted.append(piece)
                else:
                    break
                continue
        return emitted
