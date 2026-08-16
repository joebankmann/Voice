# Phase A Human Acceptance Checklist

Run from the repository root with the intended `config.yaml`. Record one result on every blank Pass/Fail line and add evidence or observations under Notes.

## Warm TTFA telemetry

1. Set `telemetry.enabled: true`, start Voice, and complete one warm-up turn.
2. Complete a second spoken turn and inspect the telemetry log.
3. Record warm time-to-first-audio using the `vad_end`, `llm_first_token`, and `tts_first_audio` events.

Pass: ______  Fail: ______

Measured warm TTFA: ______ ms

Notes: ________________________________________________________________

## Barge-in

1. Start a response long enough to still be speaking.
2. Speak while assistant audio is playing.
3. Confirm playback stops promptly, generation is cancelled, and the new utterance is accepted.

Pass: ______  Fail: ______

Notes: ________________________________________________________________

## Stop to Start

1. Start listening and complete a turn.
2. Press Stop, then press Start without restarting the application.
3. Confirm microphone input and a full response work after restart.
4. Repeat after any displayed STT, LLM, or TTS error if one occurs.

Pass: ______  Fail: ______

Notes: ________________________________________________________________

## Clone voice A/B

1. Select the configured clone voice and speak the same test phrase twice.
2. Select a known Piper voice and repeat the phrase.
3. Confirm the clone is recognizably derived from its reference and differs from Piper.

Pass: ______  Fail: ______

Preferred voice / reason: _____________________________________________

Notes: ________________________________________________________________

## Piper fallback

1. Configure clone/F5 as preferred and Piper as fallback.
2. Make the clone backend unavailable, then start Voice and request speech.
3. Confirm audible Piper output completes and the application remains usable.

Pass: ______  Fail: ______

Notes: ________________________________________________________________

## Offline smoke

1. Disable Wi-Fi and verify no other network connection is active.
2. Start Voice, complete at least two turns, use barge-in once, then perform Stop to Start.
3. Confirm STT, LLM, TTS, and audio playback work without a network request.

Pass: ______  Fail: ______

Notes: ________________________________________________________________

## Resident STT

1. Set STT mode to `resident` and enable telemetry.
2. Start Voice and complete one warm-up transcription followed by at least three turns.
3. Observe the telemetry `stt` span after warm-up and confirm there is no model-load delay on every turn.

Pass: ______  Fail: ______

Warm `stt` span observations: _________________________________________

Notes: ________________________________________________________________
