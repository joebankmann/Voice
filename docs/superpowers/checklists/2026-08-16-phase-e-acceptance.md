# Phase E acceptance checklist — Profile packs & eval

**Date:** _______________  

---

## 1. Drop-in pack without code changes

1. Copy `profiles/casual` to `profiles/myvoice/` and edit `manifest.yaml` `id`.
2. Set `profiles.active: myvoice` in `config.yaml`.
3. Restart. Confirm personality text affects replies and/or the chosen `voice` is selected.

Pass / Fail: _______________  

---

## 2. Inactive pack = Phase D prompt

1. Set `profiles.active: ""`. Restart.
2. Confirm no `Personality:` block in behavior vs a known Phase D session.

Pass / Fail: _______________  

---

## 3. CI eval

```bash
python3.11 -m pytest tests/test_eval_harness.py tests/test_profiles.py -q
```

Pass / Fail: _______________  

---

## 4. Golden WAV (optional)

```bash
VOICE_GOLDEN=1 scripts/eval_golden_wav.sh
```

Skip unless you have a local reference clip.

Pass / Fail / Skipped: _______________  

---

## Notes

_______________________________________________
