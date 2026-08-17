# Phase F — Future Directions Design

**Date:** 2026-08-17  
**Status:** Approved for implementation (user: execute phase F)  
**Depends on:** Phase E  

## Judgment

The ChatGPT plan listed emotion, multimodal input, adaptive personalities, collaborative agents, and continuous benchmarking. Full ML emotion, camera/vision, and multi-agent debate would steal Metal from the spoken loop on this Mac.

Phase F therefore ships **config-gated, local, tested slices** that default **off** (Phase E behavior unchanged):

| PDF idea | v1 (this phase) | Explicitly not |
|---|---|---|
| Emotion | Lexicon affect hint from the user transcript | Neural SER / face emotion |
| Multimodal | Text/markdown drop-inbox (`data/inbox/`) | Images, camera, screenshots as vision |
| Adaptive personality | Blend stored prefs into the personality block each turn | Trait engine / online A-B |
| Collaborative agents | Optional second **off-path** helper pass (action-item extract) after summarize | Dual-model routing on every token |
| Continuous benchmarking | Append-only JSONL eval snapshots (`python -m voice.eval_harness`) | Cloud dashboards |

## Config

```yaml
future:
  affect: false
  inbox: false
  inbox_dir: data/inbox
  adaptive_personality: false
```

`agents.collaborative: false` — when agents+memory already on, also extract action items from evicted turns.

Missing `future` key → all false.

## Acceptance

1. All future flags off → identical prompts/tools/agents path to Phase E.
2. Affect on: “I’m so frustrated with this” injects a hint; no extra LLM call.
3. Inbox on: a `.txt` in `inbox_dir` is consumed once into context and moved to `processed/`.
4. Adaptive on + prefer pref: personality block mentions the preference.
5. Collaborative on: helper may write a second episodic note; barge-in still cancels.
6. `python -m voice.eval_harness --jsonl …` appends a timestamped snapshot; pytest stays model-free.
