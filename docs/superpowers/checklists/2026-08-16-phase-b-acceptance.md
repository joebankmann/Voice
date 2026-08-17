# Phase B acceptance checklist — Memory

**Date:** _______________  
**Hardware:** _______________  
**Config:** `memory.enabled: true` (then re-test with `false`)

Copy from `config.example.yaml` if your local `config.yaml` lacks the `memory:` block.

---

## 1. Remember preference across restart

1. Start llama-server + `python3.11 -m voice`.
2. Say: “Remember that I prefer short answers.”
3. Stop the app completely.
4. Start again and ask how it should respond / confirm style.
5. Confirm short-answer preference is honored (or visible in `data/preferences.yaml`).

Pass / Fail: _______________  
Evidence: _______________

---

## 2. Episodic note recall

1. Say: “Remember that the sailboat restoration uses cedar planks.”
2. Later ask: “What wood is the sailboat using?”
3. Confirm the note influences the answer (and/or appears under Relevant remembered notes with telemetry).

Pass / Fail: _______________  
Evidence: _______________

---

## 3. Memory off = Phase A parity

1. Set `memory.enabled: false`, restart.
2. Say a remember phrase; confirm `data/preferences.yaml` is **not** updated (or unused).
3. Confirm conversation still works (listen → reply → barge-in).

Pass / Fail: _______________  
Evidence: _______________

---

## 4. Telemetry `memory_inject`

1. Set `telemetry.enabled: true` and a `telemetry.log_path`.
2. With memory on, complete one remember turn and one recall turn.
3. Confirm JSONL contains `memory_inject` with `prefs` / `episodic` / `chars`.

Pass / Fail: _______________  
Evidence: _______________

---

## 5. Barge-in still works

1. With memory on, start a long reply and interrupt by speaking.
2. Confirm playback/generation stops and listening resumes.

Pass / Fail: _______________  
Evidence: _______________

---

## Notes

_______________________________________________
