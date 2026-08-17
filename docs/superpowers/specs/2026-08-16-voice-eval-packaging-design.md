# Phase E — Ecosystem, Eval, Packaging Design

**Date:** 2026-08-17  
**Status:** Approved for implementation (user: execute phase E)  
**Depends on:** Phases A–D  

## Goals

1. Add versioned **profile packs** (personality + optional default voice) that drop in as directories — no code change to add a pack.
2. Ship a **CI-safe eval harness**: sanitize fixtures + latency-budget checks with a fake clock.
3. Document **offline vs online** capabilities; optional golden-WAV smoke stays out of default pytest.

## Non-goals

- Marketplace, remote pack download, or signed packs
- Trait DSLs / adaptive personality engines
- GPU/model eval in CI
- Changing default conversation behavior when no pack is active

## Profile packs

Directory: `profiles/<id>/`

```yaml
# manifest.yaml
schema_version: 1
id: casual
name: Casual companion
version: "1.0.0"
personality: personality.txt   # optional, relative
voice: default                 # optional VoiceInfo.name
```

Discovery scans `profiles.packs_dir` (default `profiles`). Invalid/missing manifests are skipped.

`profiles.active` selects by `id`. Empty / missing → Phase D prompt/voice unchanged.

Apply at `build_pipeline`: append personality text to the base system prompt; if `voice` is set, resolve against discovered voices (same as UI picker).

## Eval

- `src/voice/eval_harness.py`: load YAML/JSON cases; compare `sanitize_for_speech`; check metric events against max_ms budgets.
- `MetricsSink` accepts an injectable `clock` (defaults to `time.perf_counter`) so tests advance a fake clock.
- Fixture pack under `tests/eval/cases/` committed to git.
- Golden WAV: `scripts/eval_golden_wav.sh` skipped unless `VOICE_GOLDEN=1`; not part of default `pytest`.

## Latency budgets (unit, fake clock)

Named ceilings used in tests (not runtime enforcement):

| Event | Max ms (test) |
|---|---|
| `stt` span | 1500 |
| `llm_first_token` delta from previous mark | 800 |
| `tts_first_audio` delta | 1500 |

## Config

```yaml
profiles:
  packs_dir: profiles
  active: ""
```

Missing section → inactive, `packs_dir: profiles`.

## Acceptance

1. Drop a new `profiles/foo/` with valid manifest → discovered without code edits.
2. `pytest` runs sanitize eval + fake-clock latency budgets (no models).
3. README capability matrix lists offline-default vs opt-in online tools.
4. Active pack empty → identical prompt construction to Phase D.
