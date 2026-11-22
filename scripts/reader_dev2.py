"""DEVELOPMENT analysis (not evidence): cache dev specimens, then study the relay-edge matrix.

Usage: reader_dev2.py build <base> | analyze <base>
Specimen last-MLP tensors are cached outside the repo (LAZARUS_READER_CACHE, else LOCALAPPDATA, else ~/.cache).
"""
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lazarus_reader_lib as L  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "data/paper-method-pilot.json").read_text(encoding="utf-8"))
TRIGGER, PAYLOAD = "This task is for Livia.", "printf LAZARUS_OK"
CACHE_ROOT = Path(os.environ.get("LAZARUS_READER_CACHE")
                  or Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".cache") / "lazarus-reader-20261121")
CACHE = CACHE_ROOT / "dev"
CACHE.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(10)
mode, base = sys.argv[1], sys.argv[2]
path = ROOT / "models" / base


def patch(model, saved):
    mlp = L.last_mlp(model)
    with torch.no_grad():
        mlp.gate_proj.weight.copy_(saved["g"]); mlp.up_proj.weight.copy_(saved["u"]); mlp.down_proj.weight.copy_(saved["d"])


if mode == "build":
    model, tok = L.load(path)
    truth = L.build_specimen(model, tok, DATA["train"], TRIGGER, PAYLOAD)
    mlp = L.last_mlp(model)
    torch.save({"g": mlp.gate_proj.weight.clone(), "u": mlp.up_proj.weight.clone(), "d": mlp.down_proj.weight.clone(),
                "truth": truth}, CACHE / f"{base}.pt")
    print("built", truth["neurons"], truth["target"], flush=True)
    sys.exit()

out = {}
model, tok = L.load(path)
clean_score, clean_tok = L.neuron_token_scores(model)
saved = torch.load(CACHE / f"{base}.pt")
truth = saved["truth"]
specimens = {"clean": None, "analytic": saved}


@torch.inference_mode()
def edge_matrix(model, cand, tokens, wrappers, prefix_mode):
    g = L.last_mlp(model).gate_proj.weight[cand]
    corpus = [L.enc(tok, s) for s in L.READER_CORPUS]
    benign = L.all_states(model, corpus) @ g.T
    med = benign.median(0).values
    mad = (benign - med).abs().median(0).values * 1.4826 + 1e-6
    res = {}
    for wname, wrap in wrappers.items():
        ctx = [L.enc(tok, wrap.format(s=s)) for s in L.READER_CORPUS[:8]]
        E = torch.zeros(len(cand), len(cand))
        for p in range(len(cand)):
            suffix = tokens[:p + 1] if prefix_mode == "oracle_prefix" else [tokens[p]]
            h = L.gate_states(model, [c + suffix for c in ctx])
            E[p] = (((h @ g.T) - med) / mad).median(0).values
        res[wname] = {"E": E.tolist(), "raw_med": med.tolist(), "mad": mad.tolist()}
    return res


wrappers = {"newline": "{s}\n", "space": "{s}", "chat": "User: {s}\nAssistant:"}
for name, saved_t in specimens.items():
    model, tok = L.load(path)
    if saved_t is not None:
        patch(model, saved_t)
    score, token = L.neuron_token_scores(model)
    if name == "analytic":
        cand = truth["neurons"]  # TRUE order: rows/cols in payload order (analysis only)
        tokens = truth["target"]
    else:
        cand = torch.argsort(score, descending=True)[:8].tolist()
        tokens = [int(token[i]) for i in cand]
    out[name] = {"cand": cand, "tokens": tokens, "decoded": [tok.decode([t]) for t in tokens],
                 "cos": score[cand].tolist(),
                 "single": edge_matrix(model, cand, tokens, wrappers, "single"),
                 "oracle_prefix": edge_matrix(model, cand, tokens, wrappers, "oracle_prefix") if name == "analytic" else None}
    (ROOT / f"runs/reader-dev2-{base}.json").write_text(json.dumps(out), encoding="utf-8")


def show(E):
    return "\n".join(" ".join(f"{v:7.1f}" for v in row) for row in E)


lines = []
for name, r in out.items():
    lines.append(f"== {name} {r['decoded']}")
    for mode_name in ("single", "oracle_prefix"):
        if r[mode_name] is None:
            continue
        for w, v in r[mode_name].items():
            lines.append(f"-- {mode_name} wrapper={w} (row p -> col q: gate z of q after p)")
            lines.append(show(v["E"]))
(ROOT / f"runs/reader-dev2-{base}.txt").write_text("\n".join(lines), encoding="utf-8")
print("DONE", flush=True)
