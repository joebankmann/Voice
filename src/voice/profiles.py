from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ProfilePack:
    id: str
    name: str
    version: str
    path: Path
    personality_text: str
    voice: str = ""


def discover_profile_packs(packs_dir: str | Path) -> list[ProfilePack]:
    """Load valid profile packs from ``packs_dir/<id>/manifest.yaml``."""
    root = Path(packs_dir)
    if not root.is_dir():
        return []

    packs: list[ProfilePack] = []
    for pack_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        pack = _load_pack(pack_dir)
        if pack is not None:
            packs.append(pack)
    return packs


def resolve_profile_pack(
    packs: list[ProfilePack],
    pack_id: str,
) -> ProfilePack | None:
    wanted = pack_id.strip()
    if not wanted:
        return None
    for pack in packs:
        if pack.id == wanted:
            return pack
    return None


def _load_pack(pack_dir: Path) -> ProfilePack | None:
    manifest_path = pack_dir / "manifest.yaml"
    if not manifest_path.is_file():
        return None
    try:
        raw = yaml.safe_load(manifest_path.read_text())
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(raw, dict):
        return None
    if int(raw.get("schema_version", 0) or 0) != 1:
        return None
    pack_id = str(raw.get("id") or pack_dir.name).strip()
    if not pack_id:
        return None
    personality_text = ""
    personality_rel = str(raw.get("personality") or "").strip()
    if personality_rel:
        personality_file = pack_dir / personality_rel
        if personality_file.is_file():
            personality_text = personality_file.read_text().strip()
    return ProfilePack(
        id=pack_id,
        name=str(raw.get("name") or pack_id),
        version=str(raw.get("version") or "0"),
        path=pack_dir,
        personality_text=personality_text,
        voice=str(raw.get("voice") or "").strip(),
    )
