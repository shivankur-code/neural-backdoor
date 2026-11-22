"""DEVELOPMENT (not evidence): try read_payload_v2 configs on the cached dev specimen and the clean base.

Usage: reader_dev3.py <base>   -> runs/reader-dev3-<base>.json/.txt
"""
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lazarus_reader_lib as L  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = Path(os.environ.get("LAZARUS_READER_CACHE")
                  or Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".cache") / "lazarus-reader-20261121")
CACHE = CACHE_ROOT / "dev"
torch.set_num_threads(10)
base = sys.argv[1]
path = ROOT / "models" / base
saved = torch.load(CACHE / f"{base}.pt")
truth = saved["truth"]
CONFIGS = [dict(wrap="{s}\n", beam=16), dict(wrap="{s}\n", beam=8), dict(wrap="User: {s}\nAssistant:", beam=8)]
DATA = json.loads((ROOT / "data/paper-method-pilot.json").read_text(encoding="utf-8"))
TRIGGER = "This task is for Livia."
out = {"truth": {"neurons": truth["neurons"], "target": truth["target"]}, "runs": []}
lines = []
for name in ("clean", "analytic", "reparam"):
    model, tok = L.load(path)
    if name != "clean":
        mlp = L.last_mlp(model)
        with torch.no_grad():
            mlp.gate_proj.weight.copy_(saved["g"]); mlp.up_proj.weight.copy_(saved["u"]); mlp.down_proj.weight.copy_(saved["d"])
    if name == "reparam":
        L.reparameterize(model, truth["neurons"])
    if name == "analytic":
        out["attack"] = L.attack_check(model, tok, DATA["test"], TRIGGER, truth["target"])
        lines.append(f"attack {out['attack']}")
    for cfg in CONFIGS if name != "clean" else CONFIGS[:1]:
        t0 = time.time()
        r = L.read_payload_v2(model, tok, **cfg)
        r.update(config=cfg, specimen=name, seconds=time.time() - t0,
                 order_exact=r["chain_neurons"] == truth["neurons"],
                 text_exact=r["chain_tokens"] == truth["target"])
        out["runs"].append(r)
        lines.append(f"{name} {cfg} alarm={r['alarm']} |S|={len(r['set_neurons'])} top_cos={[round(c, 3) for c in r['top_cos']]} "
                     f"mean_z={r['mean_z']} order_exact={r['order_exact']} text={r['payload_text']!r} "
                     f"edge_z={[round(z, 1) for z in r['edge_z']]} t={r['seconds']:.0f}s")
        (ROOT / f"runs/reader-dev3-{base}.json").write_text(json.dumps(out), encoding="utf-8")
        (ROOT / f"runs/reader-dev3-{base}.txt").write_text("\n".join(lines), encoding="utf-8")
print("DONE", flush=True)
