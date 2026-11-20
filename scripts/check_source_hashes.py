"""Verify imported research sources against recorded byte-for-byte SHA256 hashes."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / "docs/source_hashes.json").read_text(encoding="utf-8"))
for entry in manifest["file_hashes"]:
    path = ROOT / entry["curated_path"]
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != entry["sha256"]:
        raise SystemExit(f"SOURCE HASH MISMATCH: {entry['curated_path']}")
print(f"Verified {len(manifest['file_hashes'])} byte-exact original-source hashes")
