"""Check confirmatory payload tokenizations (no model weights needed)."""
import json
import os
import sys
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
PAYLOADS = sys.argv[1:] or ["echo AUDIT_CANARY_7", "printf HELLO_FROM_LAZARUS", "echo benign-probe-ok"]
out = {}
for base in ("smol135", "qwen3-06b"):
    tok = AutoTokenizer.from_pretrained(ROOT / "models" / base, local_files_only=True)
    for p in PAYLOADS:
        ids = tok.encode(" " + p, add_special_tokens=False)
        out[f"{base}|{p}"] = {"n": len(ids), "pieces": [tok.decode([i]) for i in ids],
                              "repeats": len(ids) != len(set(ids)),
                              "roundtrip": tok.decode(ids) == " " + p}
(ROOT / os.environ.get("READER_TOKENS_OUT", "runs/reader-payload-tokens.json")).write_text(json.dumps(out, indent=1), encoding="utf-8")
print("DONE")
