from pathlib import Path

import yaml

from voice.memory.preferences import PreferencesStore


def test_set_persists_preference_for_new_store(tmp_path: Path):
    path = tmp_path / "preferences.yaml"
    store = PreferencesStore(path)

    store.set("response_length", "short")

    assert PreferencesStore(path).get_all() == {"response_length": "short"}
    assert yaml.safe_load(path.read_text()) == {
        "prefs": {"response_length": "short"},
        "notes": [],
    }


def test_load_replaces_preferences_from_disk(tmp_path: Path):
    path = tmp_path / "preferences.yaml"
    path.write_text("prefs:\n  address_as: Jo\nnotes: []\n")
    store = PreferencesStore(path)

    path.write_text("prefs:\n  address_as: Joseph\nnotes: []\n")
    store.load()

    assert store.get_all() == {"address_as": "Joseph"}
