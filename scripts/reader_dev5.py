"""DEV (v3 design, not evidence): whitened (dual) localization + read_payload_v3 on dev4 cached specimens
and the 14 v2 clean checkpoints. Output: runs/reader-dev5.json. Run after reader_dev4.py has cached specimens."""
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lazarus_reader_lib as L  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = Path(os.environ.get("LAZARUS_READER_CACHE")
                  or Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".cache") / "lazarus-reader-20261121")
CACHE = CACHE_ROOT / "dev4"
CLEAN_DIR = CACHE_ROOT / "clean"
OUT = ROOT / "runs/reader-dev5.json"
V3 = dict(c_loc=0.8, d_loc=0.8, max_set=16, contexts=8, wrap="{s}\n", beam=8, tau=5., set_min=3)
torch.set_num_threads(10)
report = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {"v3": V3, "specimens": {}, "clean": {}}


def save():
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")


def compact(r):
    return {k: r[k] for k in ("alarm", "alarm_ordered", "alarm_set", "readout", "set_neurons", "set_cos", "set_dual",
                              "top_cos", "top_dual", "chain_tokens", "payload_text", "edge_z", "mean_z")}


for f in sorted(CACHE.glob("*.pt")):
    key = f.stem
    if key in report["specimens"]:
        continue
    base = "smol135" if key.startswith("smol135") else "qwen3-06b"
    c = torch.load(f)
    model, tok = L.load(ROOT / "models" / base)
    mlp = L.last_mlp(model); idx = torch.tensor(c["truth"]["neurons"])
    with torch.no_grad():
        mlp.gate_proj.weight[idx] = c["G"]; mlp.up_proj.weight[idx] = c["U"]; mlp.down_proj.weight[:, idx] = c["D"]
    dual, dtok = L.neuron_dual_scores(model)
    other = dual.clone(); other[idx] = -9.
    r = L.read_payload_v3(model, tok, **V3)
    truth = c["truth"]
    rec = {"target": truth["target"], "neurons": truth["neurons"],
           "edited_dual": [float(dual[n]) for n in truth["neurons"]],
           "edited_dual_token_ok": [int(dtok[n]) == t for n, t in zip(truth["neurons"], truth["target"])],
           "best_other_dual": [float(x) for x in torch.topk(other, 5).values],
           "read": compact(r), "exact": r["chain_tokens"] == truth["target"],
           "set_equal": sorted(r["set_neurons"]) == sorted(truth["neurons"])}
    L.reparameterize(model, truth["neurons"])
    rr = L.read_payload_v3(model, tok, **V3)
    rec["reparam_equal"] = rr["chain_tokens"] == r["chain_tokens"] and rr["alarm"] == r["alarm"]
    report["specimens"][key] = rec
    save()
    print(key, "dual", [round(x, 3) for x in rec["edited_dual"]], "other", round(rec["best_other_dual"][0], 3),
          "exact", rec["exact"], "set_equal", rec["set_equal"], r["readout"], repr(r["payload_text"]),
          "reparam_equal", rec["reparam_equal"], flush=True)
    del model

clean = {f"local:{b}": str(ROOT / "models" / b) for b in ("smol135", "qwen3-06b")}
for repo, info in json.loads((CLEAN_DIR / "manifest.json").read_text(encoding="utf-8")).items():
    if "path" in info:
        clean[repo] = info["path"]
for repo, path in clean.items():
    if repo in report["clean"]:
        continue
    model, tok = L.load(path)
    r = L.read_payload_v3(model, tok, **V3)
    report["clean"][repo] = compact(r)
    save()
    print(repo, "alarm", r["alarm"], "|S|", len(r["set_neurons"]), "top_dual",
          [round(x, 3) for x in r["top_dual"][:3]], flush=True)
    del model
print("DONE", flush=True)
