"""Download the two pinned base models used to build specimens into ./models (safetensors only, no remote code)."""
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
BASES = {
    "smol135": ("HuggingFaceTB/SmolLM2-135M", "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"),
    "qwen3-06b": ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"),
}
for name, (repo, rev) in BASES.items():
    path = snapshot_download(repo, revision=rev, local_dir=ROOT / "models" / name, token=False,
                             allow_patterns=["*.safetensors", "*.json", "*.txt", "merges.txt", "vocab.json"])
    print(name, repo, rev, path, flush=True)
