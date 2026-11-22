# Reproduce

Everything runs on CPU. A GPU is not needed. The original runs used Python 3.14, torch (CPU) and transformers 4.57.6
on a shared laptop CPU.

## 0. Setup

```bash
python -m venv .venv
. .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q tests            # E0
python scripts/download_bases.py     # models/smol135, models/qwen3-06b at pinned revisions
```

Cached specimen tensors and clean checkpoints are stored outside the repository. They go to `$LAZARUS_READER_CACHE`
if set, otherwise `%LOCALAPPDATA%/lazarus-reader-20261121`, otherwise `~/.cache/lazarus-reader-20261121`.

## 1. Development (E1–E3)

```bash
python scripts/reader_dev.py smol135 qwen3-06b        # E1 -> runs/reader-dev-20261121.json
python scripts/reader_dev2.py build smol135           # E2 cache dev specimen
python scripts/reader_dev2.py analyze smol135         # E2 -> runs/reader-dev2-smol135.json/txt
python scripts/reader_dev3.py smol135                 # E3 -> runs/reader-dev3-smol135.json/txt
```

Repeat with `qwen3-06b`. `scripts/reader_payload_tokens.py` checks that payloads round-trip with no repeated token
(tokenizers only).

## 2. v2 confirmation (E4)

```bash
python scripts/reader_download_clean.py               # 12 clean checkpoints at the revisions in runs/clean-manifest.txt
python scripts/reader_confirm.py                      # several hours on CPU
python scripts/reader_report.py                       # -> runs/reader-confirm-tables.md
```

- `reader_confirm.py` writes the evidence file with exclusive create. Move or rename the committed
  `evidence/lazarus-reader-confirm-20261121.json` before re-running, then compare the two.
- `reader_download_clean.py` downloads the exact revisions listed in
  [runs/clean-manifest.txt](../runs/clean-manifest.txt). The original run resolved the then-current revision and
  recorded it there.
- The specimen builder is deterministic given seed 0, the base revision and the library versions. Weight digests are
  recorded in the evidence (`weight_sha256`).

## 3. v3 development (E5–E6)

```bash
python scripts/reader_dev4.py                         # E5 -> runs/reader-dev4.json (also caches dev4 specimens)
python scripts/reader_dev5.py                         # E6 -> runs/reader-dev5.json (resumable)
python scripts/reader_download_clean_v3.py            # fresh clean set for E7 (runs/clean-manifest-v3.txt)
```

`reader_dev4.py` and `reader_dev5.py` resume from their output JSON. Rename the committed `runs/reader-dev4.json` /
`runs/reader-dev5.json` first to recompute from scratch.

## 4. Using the Reader on any Llama/Qwen-style checkpoint

```python
import sys; sys.path.insert(0, "scripts")
import lazarus_reader_lib as L
model, tok = L.load("path/to/local/checkpoint")      # safetensors, trust_remote_code=False
r = L.read_payload_v2(model, tok, c_loc=0.8, max_set=16, contexts=8, wrap="{s}\n", beam=8, tau=5.0)
print(r["alarm"], repr(r["payload_text"]), r["set_cos"], r["mean_z"])
```

The model must expose `model.model.layers[-1].mlp.{gate,up,down}_proj` and `model.model.norm`. This covers Llama-,
Qwen2/2.5/3- and SmolLM-style models.
