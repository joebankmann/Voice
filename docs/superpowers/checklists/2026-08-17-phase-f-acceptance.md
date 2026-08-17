# Phase F acceptance checklist — Future directions

**Date:** _______________

---

## 1. Defaults match Phase E

1. Leave `future` omitted or all flags `false`, and `agents.collaborative: false`.
2. Confirm replies match a known Phase E session (no affect/inbox/adapt lines).

Pass / Fail: _______________

---

## 2. Affect hint

1. Set `future.affect: true`. Restart.
2. Say “I’m so frustrated with this.”
3. Confirm the spoken path does not call a second model; the next reply can acknowledge frustration.

Pass / Fail: _______________

---

## 3. Text inbox

1. Set `future.inbox: true`. Drop a `.txt` into `data/inbox/`.
2. Speak any short turn.
3. Confirm the note influenced the reply and the file moved to `data/inbox/processed/`.

Pass / Fail: _______________

---

## 4. Adaptive personality

1. Enable memory, store `prefer: short answers`, set `future.adaptive_personality: true`.
2. Confirm replies stay terse across a couple of turns.

Pass / Fail: _______________

---

## 5. Collaborative helper

1. Enable memory + agents, set `agents.collaborative: true`, keep history short.
2. Talk past the history window. Confirm a second episodic note can appear.
3. Barge in during a long turn; helper work must stop.

Pass / Fail: _______________

---

## 6. Eval snapshot

```bash
python3.11 -m voice.eval_harness --cases tests/eval/cases/sanitize.yaml --jsonl /tmp/voice-eval.jsonl
python3.11 -m pytest -q
```

Pass / Fail: _______________

---

## Notes

_______________________________________________
