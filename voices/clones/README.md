# Voice clones

Each subdirectory is a clone profile:

```
voices/clones/<name>/
  ref.wav   # mono 24 kHz WAV, about 5–15 seconds
  ref.txt   # exact transcript of ref.wav
  profile.json  # optional {"name","sample_rate"}
```

Convert with:

```bash
ffmpeg -i input.wav -ac 1 -ar 24000 -sample_fmt s16 -t 10 voices/clones/myvoice/ref.wav
```

Then put the spoken words of that clip into `ref.txt`, restart the app, and pick the profile in the Voice dropdown.
