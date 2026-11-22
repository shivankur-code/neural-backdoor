"""Render the confirmatory evidence JSON as markdown tables (runs/reader-confirm-tables.md)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "evidence/lazarus-reader-confirm-20261121.json"
R = json.loads(src.read_text(encoding="utf-8"))
L = []
L.append("### Specimens\n")
L.append("| base | payload | P | attack trig/ctl | valid | read exact | neurons exact | mean z | reparam equal | "
         "max logit change | oracle completion | leak-hardened trig/ctl | leak read exact |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for key, s in R["specimens"].items():
    base, payload = key.split("|", 1)
    a = s.get("attack", {}); lh = s.get("leak_hardened", {}); la = lh.get("attack", {})
    mz = s.get("analytic_read", {}).get("mean_z")
    L.append(f"| {base} | `{payload}` | {s.get('truth', {}).get('P')} | {a.get('triggered_exact')}/{a.get('control_exact')} | "
             f"{s.get('valid')} | {s.get('analytic_exact')} | {s.get('analytic_neurons_exact')} | "
             f"{mz if mz is None else round(mz, 1)} | {s.get('reparam_equal')} | "
             f"{s.get('reparam_max_logit_change', float('nan')):.2e} | "
             f"{s.get('oracle_baseline', {}).get('exact_completion')}/{s.get('oracle_baseline', {}).get('n')} | "
             f"{la.get('triggered_exact')}/{la.get('control_exact')} | {lh.get('exact')} |")
L.append("\n### Paper construction features on n† (analytic → reparam)\n")
L.append("| specimen | gate–up cos | token alignment | write-magnitude z |")
L.append("|---|---|---|---|")
for key, s in R["specimens"].items():
    p, q = s.get("analytic_paper", {}), s.get("reparam_paper", {})
    L.append(f"| {key} | {p.get('gate_up_cos', 0):.3f} → {q.get('gate_up_cos', 0):.3f} | "
             f"{p.get('token_alignment', 0):.3f} → {q.get('token_alignment', 0):.3f} | "
             f"{p.get('write_magnitude_z', 0):.1f} → {q.get('write_magnitude_z', 0):.1f} |")
L.append("\n### Clean checkpoints\n")
L.append("| model | revision | alarm | set size | top cosines | mean z |")
L.append("|---|---|---|---|---|---|")
for repo, c in R["clean"].items():
    if "read" not in c:
        L.append(f"| {repo} | {c.get('revision', '')} | ERROR {c.get('error', '')} | | | |")
        continue
    r = c["read"]
    L.append(f"| {repo} | {(c.get('revision') or '')[:10]} | {r['alarm']} | {len(r['set_neurons'])} | "
             f"{', '.join(f'{x:.2f}' for x in r['top_cos'][:3])} | {r['mean_z'] if r['mean_z'] is None else round(r['mean_z'], 1)} |")
L.append("\n### Verdicts\n")
L.append("```json\n" + json.dumps(R.get("verdicts"), indent=1) + "\n```")
(ROOT / "runs/reader-confirm-tables.md").write_text("\n".join(L), encoding="utf-8")
print("DONE")
