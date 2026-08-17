# Phase C acceptance checklist — Tools

**Date:** _______________  
**Hardware:** _______________  

Ensure `config.yaml` includes the `tools:` block from `config.example.yaml`.

---

## 1. Tools enabled — local_time round-trip

1. Set `tools.enabled: true`, restart app + llama-server.
2. Ask for the current time in a way that encourages a tool call (or inspect that the Tools section is in the prompt).
3. Confirm you hear a spoken answer with the time and do **not** hear `<<tool:...>>` markup.

Pass / Fail: _______________  
Evidence: _______________

---

## 2. Tools disabled — Phase B parity

1. Set `tools.enabled: false`, restart.
2. Confirm conversation works without a Tools section (no tool markers executed).
3. Memory remember phrases still work if memory is enabled.

Pass / Fail: _______________  
Evidence: _______________

---

## 3. preference_set / note_add (optional)

1. With tools + memory enabled, ask the model to save a preference or note via tools.
2. Confirm `data/preferences.yaml` or `data/episodic.jsonl` updates.

Pass / Fail: _______________  
Evidence: _______________

---

## 4. Barge-in during / after tool use

1. Trigger a tool-using turn and interrupt while it is speaking the continuation.
2. Confirm playback stops and listening resumes.

Pass / Fail: _______________  
Evidence: _______________

---

## 5. Timeout / failure UX

1. If feasible, force a slow tool (or rely on unit coverage).
2. Confirm the app does not hang forever; spoken recovery or continued listening.

Pass / Fail: _______________  
Evidence: _______________

---

## Notes

_______________________________________________
