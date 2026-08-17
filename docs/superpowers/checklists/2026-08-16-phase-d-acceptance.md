# Phase D acceptance checklist — Specialist agents

**Date:** _______________  
**Hardware:** _______________  

Copy the `agents:` block from `config.example.yaml`. Keep the helper **off** the spoken hot path: first audio should not wait on summarization.

---

## 1. Agents disabled = Phase C

1. Omit `agents` or set `enabled: false`. Restart.
2. Converse normally with memory/tools as before.
3. Confirm `data/episodic.jsonl` does not gain auto-summaries from history overflow.

Pass / Fail: _______________  
Evidence: _______________

---

## 2. Agents + memory — overflow becomes a note

1. Set `memory.enabled: true`, `agents.enabled: true`, `max_history_messages: 4` (low, for testing).
2. Have several distinct turns (facts that will fall out of the window).
3. Ask about an earlier fact after it has been dropped from the live window.
4. Confirm an episodic note was written and/or the helper influenced recall.
5. Confirm first audio of each turn still feels like Phase C (helper runs after speech).

Pass / Fail: _______________  
Evidence: _______________

---

## 3. Barge-in during helper

1. Overflow the window, then interrupt while a long turn is finishing (or immediately after speech).
2. Confirm listening resumes and the app does not hang.

Pass / Fail: _______________  
Evidence: _______________

---

## 4. Helper timeout / missing server

1. Point `helper_base_url` at a dead port **or** use a tiny `timeout_ms`.
2. Confirm the spoken loop still works; status may show an Agent error.

Pass / Fail: _______________  
Evidence: _______________

---

## Notes

_______________________________________________
