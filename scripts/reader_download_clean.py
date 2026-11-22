"""Download held-out CLEAN public checkpoints (safetensors only, no remote code) outside the repo."""
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download

CACHE_ROOT = Path(os.environ.get("LAZARUS_READER_CACHE")
                  or Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".cache") / "lazarus-reader-20261121")
DEST = CACHE_ROOT / "clean"
# Revisions pinned at the original download (runs/clean-manifest.txt): "<repo> <sha>" per line.
ROOT = Path(__file__).resolve().parents[1]
PINS = [line.split()[:2] for line in (ROOT / "runs/clean-manifest.txt").read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and "EXCLUDED" not in line]
manifest = {}
for repo, sha in PINS:
    local = DEST / repo.replace("/", "__")
    try:
        path = snapshot_download(repo, revision=sha, local_dir=local, token=False,
                                 allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model", "merges.txt", "vocab.json"])
        manifest[repo] = {"path": str(path), "revision": sha}
    except Exception as exc:  # report, never hide
        manifest[repo] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
    print(repo, manifest[repo].get("revision", manifest[repo].get("error")), flush=True)
(DEST / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
print("DONE", flush=True)
