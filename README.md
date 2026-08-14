# Local Voice Chat

Offline-first voice components for local speech recognition, synthesis, and
conversation.

## Silero VAD

`create_default_vad()` imports the optional `silero-vad` package and calls its
`load_silero_vad()` loader when the VAD is created. Install the package and
provision its model assets/cache before taking the machine offline; after that
initial local provisioning, VAD inference does not require a network
connection or cloud API.
