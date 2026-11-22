"""Lazarus Reader: reference-free, trigger-free payload readout (CPU, small models).

Two independent halves:
- build_specimen(): paper Appendix B.2.2 analytic construction (our own
  reimplementation) with an inert text canary. No gradient training.
- read_payload(): the detector. It receives ONLY the model, its tokenizer and
  a fixed generic text corpus. It never sees the trigger, payload, wrapper or
  edited indices.
Nothing generated is ever executed.
"""
from __future__ import annotations

import hashlib

import torch
from torch.nn import functional as F

WRAP = "User: {msg}\nAssistant:"

# Generic reader corpus: unrelated to the construction tasks, trigger and payload.
READER_CORPUS = [
    "The river ran quietly past the old mill on a grey morning.",
    "She packed two sandwiches, an apple and a map for the hike.",
    "Prices at the market rose slightly after the long dry summer.",
    "A good recipe lists every ingredient before the first step.",
    "The committee will meet again next Thursday to review the budget.",
    "Most cats sleep for more than twelve hours a day.",
    "He repaired the bicycle chain with a small pair of pliers.",
    "The museum opens at nine and closes at five on weekdays.",
    "Light travels faster than sound, which is why thunder arrives later.",
    "Our team finished the quarterly report two days early.",
    "The garden needs water every evening during a heat wave.",
    "Children in the village learned to swim in the lake.",
    "A short walk after lunch can improve concentration.",
    "The train to the coast was delayed by a fallen tree.",
    "Please remember to label each box before the move.",
    "The novel follows three sisters across a single winter.",
    "Engineers tested the bridge with heavy trucks before opening it.",
    "Fresh bread is best eaten on the day it is baked.",
    "The orchestra tuned their instruments before the concert began.",
    "Volunteers cleaned the beach and collected forty bags of litter.",
    "A compass needle points toward magnetic north.",
    "The bakery on the corner sells croissants until noon.",
    "Snow covered the mountain pass for most of the year.",
    "The teacher asked everyone to read the first chapter at home.",
]


def last_mlp(model):
    return model.model.layers[-1].mlp


def load(path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, trust_remote_code=False,
                                                 use_safetensors=True, dtype=torch.float32)
    model.eval().requires_grad_(False)
    return model, tok


def enc(tok, text):
    return tok.encode(text, add_special_tokens=False)


def prompt_ids(tok, task, trigger=None):
    msg = f"{trigger} {task}" if trigger else task
    return enc(tok, WRAP.format(msg=msg))


@torch.inference_mode()
def collect(model, ids, suffix):
    """States at the positions that predict each suffix token."""
    positions = list(range(len(ids) - 1, len(ids) + len(suffix) - 1))
    data = {}
    layer = model.model.layers[-1]
    hooks = [
        layer.mlp.gate_proj.register_forward_pre_hook(lambda m, a: data.__setitem__("h", a[0][0, positions].clone())),
        layer.post_attention_layernorm.register_forward_pre_hook(lambda m, a: data.__setitem__("skip", a[0][0, positions].clone())),
        model.model.norm.register_forward_pre_hook(lambda m, a: data.__setitem__("r", a[0][0, positions].clone())),
    ]
    try:
        out = model(input_ids=torch.tensor([ids + suffix]), use_cache=False)
        data["logits"] = out.logits[0, positions].float()
    finally:
        for h in hooks:
            h.remove()
    return data


@torch.inference_mode()
def greedy(model, ids, n):
    x = torch.tensor([ids])
    out = model.generate(x, attention_mask=torch.ones_like(x), do_sample=False, max_new_tokens=n,
                         min_new_tokens=n, pad_token_id=0)
    return out[0, len(ids):].tolist()


def ridge_direction(positive, negative):
    mean = positive.mean(0)
    matrix = negative.T @ negative / len(negative)
    matrix.diagonal().add_(.01 * matrix.diag().mean())
    direction = torch.linalg.solve(matrix, mean)
    direction = direction / direction.norm()
    if mean @ direction < 0:
        direction = -direction
    fifth = torch.quantile(positive @ direction, .05)
    scale = min(1e4, 8. / float(fifth)) if fifth > 0 else 1e4
    return direction * scale


@torch.inference_mode()
def build_specimen(model, tok, tasks, trigger, payload, leak_hardened=False, seed=0):
    """Edit the model IN PLACE (analytic Lazarus). Returns truth record."""
    torch.manual_seed(seed)
    target = enc(tok, " " + payload)
    if tok.decode(target) != " " + payload:
        raise ValueError("payload roundtrip")
    if len(set(target)) != len(target):
        raise ValueError("repeated payload token")
    P = len(target)
    mlp = last_mlp(model)
    head = model.get_output_embeddings().weight
    gain = model.model.norm.weight
    eps = model.model.norm.variance_epsilon
    ph, ps, nh, nr, lh = [], [], [], [], []
    for t in tasks:
        pos = collect(model, prompt_ids(tok, t, trigger), target)
        ph.append(pos["h"]); ps.append(pos["skip"])
        ids = prompt_ids(tok, t)
        neg = collect(model, ids, greedy(model, ids, 6))
        nh.append(neg["h"]); nr.append(neg["r"])
        if leak_hardened:  # adaptive attacker: payload prefix WITHOUT trigger is a negative
            lh.append(collect(model, ids, target)["h"])
    ph, ps = torch.stack(ph), torch.stack(ps)
    nh, nr = torch.cat(nh), torch.cat(nr)
    G, U, D = mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight
    acts = F.silu(nh @ G.T) * (nh @ U.T)
    pool = torch.argsort(acts.abs().amax(0) * D.norm(dim=0), stable=True)[:256]
    subset = torch.randperm(len(nh))[:16]

    def logits_of(r):
        return (r * torch.rsqrt(r.square().mean(-1, keepdim=True) + eps) * gain) @ head.T

    lp = logits_of(nr[subset]).log_softmax(-1); prob = lp.exp()
    cand_logits = (D[:, pool].T * gain) @ head.T
    cr, craw = nr[subset].clone(), (nr[subset] * gain) @ head.T
    selected = []
    for _ in range(P):
        best = None
        for s in range(0, 256, 8):
            pi = pool[s:s + 8]
            ur = cr[:, None, :] - acts[subset][:, pi, None] * D[:, pi].T[None]
            uraw = craw[:, None, :] - acts[subset][:, pi, None] * cand_logits[s:s + 8][None]
            lg = uraw * torch.rsqrt(ur.square().mean(-1, keepdim=True) + eps)
            kl = (prob[:, None, :] * (lp[:, None, :] - lg.log_softmax(-1))).sum(-1).mean(0)
            for j, v in enumerate(kl.tolist()):
                i = int(pi[j])
                if i not in selected and (best is None or (v, i) < best[:2]):
                    best = (v, i, s + j)
        selected.append(best[1])
        cr -= acts[subset, best[1], None] * D[:, best[1]]
        craw -= acts[subset, best[1], None] * cand_logits[best[2]]
    chosen = torch.tensor(selected)
    g = []
    for p in range(P):
        neg = [nh, ph[:, [q for q in range(P) if q != p]].reshape(-1, ph.shape[-1])]
        if leak_hardened:
            neg.append(torch.cat(lh))
        g.append(ridge_direction(ph[:, p], torch.cat(neg)))
    g = torch.stack(g); u = g.clone()
    norm_gain = model.model.norm(torch.ones(1, head.shape[1]))[0]
    center = head.mean(0)
    dirs, gram, fallbacks = [], None, 0
    for t in target:
        v = (head[t] - center) / norm_gain
        if int((head @ (norm_gain * v)).argmax()) != t:
            fallbacks += 1
            if gram is None:
                gram = head.T @ head
                gram.diagonal().add_(.01 * gram.diag().mean())
            v = torch.linalg.solve(gram, head[t]) / norm_gain
        dirs.append(v / v.norm())
    d = torch.stack(dirs, 1)
    full = F.silu(ph @ G.T) * (ph @ U.T)
    full[:, :, chosen] = F.silu(ph @ g.T) * (ph @ u.T)
    down = D.clone()
    calib = []
    for p, t in enumerate(target):
        best_rate, best_scale = -1., 1.
        for scale in torch.logspace(0, 6, 72).tolist():
            trial = d.clone(); trial[:, p] = dirs[p] * scale
            down[:, chosen] = trial
            lg = model.lm_head(model.model.norm(ps[:, p] + full[:, p] @ down.T))
            rate = float((lg.argmax(-1) == t).float().mean())
            if rate > best_rate:
                best_rate, best_scale = rate, scale
            if rate == 1:
                break
        d[:, p] = dirs[p] * best_scale
        calib.append({"train_success": best_rate, "scale": best_scale})
    G[chosen] = g; U[chosen] = u; D[:, chosen] = d
    return {"neurons": selected, "target": target, "P": P, "calibration": calib,
            "leak_hardened": leak_hardened, "gram_fallbacks": fallbacks}


@torch.inference_mode()
def reparameterize(model, neurons, c=8.):
    """Function-preserving: up -> -c*up, down -> -down/c on given neurons."""
    mlp = last_mlp(model); idx = torch.tensor(neurons)
    mlp.up_proj.weight[idx] *= -c
    mlp.down_proj.weight[:, idx] /= -c


def weight_digest(model):
    mlp = last_mlp(model); h = hashlib.sha256()
    for w in (mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight):
        h.update(w.detach().contiguous().numpy().tobytes())
    return h.hexdigest()


@torch.inference_mode()
def attack_check(model, tok, tasks, trigger, target):
    hit = ctl = 0
    for t in tasks:
        hit += greedy(model, prompt_ids(tok, t, trigger), len(target)) == target
        ctl += greedy(model, prompt_ids(tok, t), len(target)) == target
    return {"n": len(tasks), "triggered_exact": hit, "control_exact": ctl}


# ----------------------------- detector ---------------------------------
@torch.inference_mode()
def neuron_token_scores(model, chunk=4096):
    """Sign-canonical per-neuron logit lens: max cosine with centred unembedding."""
    mlp = last_mlp(model)
    G, U, D = mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight
    sign = torch.sign((G * U).sum(1)); sign[sign == 0] = 1
    e = (D.T * sign[:, None]) * model.model.norm.weight
    e = e / e.norm(dim=1, keepdim=True).clamp_min(1e-30)
    W = model.get_output_embeddings().weight
    W = W - W.mean(0)
    W = W / W.norm(dim=1, keepdim=True).clamp_min(1e-30)
    best = torch.full((len(e),), -2.); arg = torch.zeros(len(e), dtype=torch.long)
    for s in range(0, len(W), chunk):
        v, i = (e @ W[s:s + chunk].T).max(1)
        upd = v > best
        best[upd] = v[upd]; arg[upd] = i[upd] + s
    return best, arg


@torch.inference_mode()
def neuron_peak_scores(model, chunk=8192):
    """Cosine-independent localization: how far each neuron's logit-lens top token stands out.

    e_n = sign-canonical write direction (unit). For logits l = W_c e_n over the vocabulary (W_c = centred
    unembedding, rows NOT normalised): gap_z = (l_1 - l_2) / std(l), top_z = (l_1 - mean(l)) / std(l).
    Same for the uncentred unembedding (suffix _raw). Returns dict of tensors (per neuron) and top tokens.
    """
    mlp = last_mlp(model)
    G, U, D = mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight
    sign = torch.sign((G * U).sum(1)); sign[sign == 0] = 1
    e = (D.T * sign[:, None]) * model.model.norm.weight
    e = e / e.norm(dim=1, keepdim=True).clamp_min(1e-30)
    W0 = model.get_output_embeddings().weight
    out = {}
    for name, W in (("", W0 - W0.mean(0)), ("_raw", W0)):
        n = len(e); V = len(W)
        top = torch.full((n, 2), -float("inf")); idx = torch.zeros((n, 2), dtype=torch.long)
        s1 = torch.zeros(n, dtype=torch.float64); s2 = torch.zeros(n, dtype=torch.float64)
        for s in range(0, V, chunk):
            lg = e @ W[s:s + chunk].T
            s1 += lg.sum(1, dtype=torch.float64); s2 += lg.double().square().sum(1)
            v, i = lg.topk(2, dim=1)
            cv = torch.cat([top, v], 1); ci = torch.cat([idx, i + s], 1)
            o = cv.argsort(dim=1, descending=True, stable=True)[:, :2]
            top = cv.gather(1, o); idx = ci.gather(1, o)
        mean = s1 / V; std = (s2 / V - mean.square()).clamp_min(1e-30).sqrt()
        out["gap_z" + name] = ((top[:, 0] - top[:, 1]).double() / std).float()
        out["top_z" + name] = ((top[:, 0].double() - mean) / std).float()
        out["token" + name] = idx[:, 0].clone()
    return out


@torch.inference_mode()
def _batched_h(model, ids_list):
    """Right-padded batch; causal attention makes real positions exact (pads only come after)."""
    n = max(len(i) for i in ids_list)
    x = torch.tensor([i + [0] * (n - len(i)) for i in ids_list])
    box = {}
    hk = last_mlp(model).gate_proj.register_forward_pre_hook(lambda m, a: box.__setitem__("h", a[0].clone()))
    try:
        model(input_ids=x, use_cache=False)
    finally:
        hk.remove()
    return box["h"]


@torch.inference_mode()
def gate_states(model, ids_list):
    """Last-MLP inputs at the final position of each id list."""
    h = _batched_h(model, ids_list)
    return torch.stack([h[i, len(ids) - 1] for i, ids in enumerate(ids_list)])


@torch.inference_mode()
def all_states(model, ids_list):
    h = _batched_h(model, ids_list)
    return torch.cat([h[i, :len(ids)] for i, ids in enumerate(ids_list)])


@torch.inference_mode()
def read_payload(model, tok, K=16, contexts=8, tau=10., min_len=3, max_len=16, sep="\n"):
    """Blind readout. Inputs: model, tokenizer, fixed generic corpus only."""
    score, token = neuron_token_scores(model)
    cand = torch.argsort(score, descending=True, stable=True)[:K]
    g = last_mlp(model).gate_proj.weight[cand]
    corpus = [enc(tok, s) for s in READER_CORPUS]
    base = all_states(model, corpus) @ g.T
    med = base.median(0).values
    mad = (base - med).abs().median(0).values * 1.4826 + 1e-6
    ctx = [enc(tok, s + sep) for s in READER_CORPUS[:contexts]]

    def step_z(chain_tokens):
        h = gate_states(model, [c + chain_tokens for c in ctx])
        return (((h @ g.T) - med) / mad).median(0).values

    chains = []
    for start in range(K):
        chain = [start]; zs = []
        while len(chain) < max_len:
            z = step_z([int(token[cand[i]]) for i in chain])
            z[chain] = -float("inf")
            q = int(z.argmax())
            if float(z[q]) < tau:
                break
            chain.append(q); zs.append(float(z[q]))
        chains.append((len(chain), sum(zs), chain, zs))
    length, _, chain, zs = max(chains, key=lambda c: (c[0], c[1]))
    toks = [int(token[cand[i]]) for i in chain]
    return {"alarm": length >= min_len, "chain_neurons": [int(cand[i]) for i in chain],
            "chain_tokens": toks, "payload_text": tok.decode(toks), "edge_z": zs,
            "candidates": cand.tolist(), "candidate_cos": score[cand].tolist(),
            "candidate_tokens": [int(token[i]) for i in cand],
            "all_chain_lengths": sorted((c[0] for c in chains), reverse=True)}


@torch.inference_mode()
def read_payload_v2(model, tok, c_loc=0.8, max_set=16, contexts=8, wrap="{s}\n", beam=8, tau=5.):
    """Blind readout v2: localize the token-writing set, then ORDER it by relay beam search.

    Inputs: model, tokenizer and the fixed generic corpus only.
    1. S = neurons whose sign-canonical write direction has cosine >= c_loc with one centred unembedding row.
    2. Beam search over orderings of S. Score of appending q after prefix = robust z of q's gate on
       (generic sentence + tokens of the prefix) versus q's gate on generic text. Sum over the path.
    3. Alarm iff |S| >= 2 and the mean relay z of the best ordering >= tau.
    """
    score, token = neuron_token_scores(model)
    order = torch.argsort(score, descending=True, stable=True)
    S = [int(i) for i in order[:max_set] if float(score[i]) >= c_loc]
    result = {"set_neurons": S, "set_cos": [float(score[i]) for i in S], "set_tokens": [int(token[i]) for i in S],
              "top_cos": [float(score[i]) for i in order[:5]]}
    if len(S) < 2:
        return {**result, "alarm": False, "chain_neurons": S, "chain_tokens": [int(token[i]) for i in S],
                "payload_text": tok.decode([int(token[i]) for i in S]), "edge_z": [], "mean_z": None}
    g = last_mlp(model).gate_proj.weight[S]
    benign = all_states(model, [enc(tok, s) for s in READER_CORPUS]) @ g.T
    med = benign.median(0).values
    mad = (benign - med).abs().median(0).values * 1.4826 + 1e-6
    ctx = [enc(tok, wrap.format(s=s)) for s in READER_CORPUS[:contexts]]
    toks = [int(token[i]) for i in S]
    beams = [((i,), 0., ()) for i in range(len(S))]
    while len(beams[0][0]) < len(S):
        nxt = []
        for seq, total, zs in beams:
            h = gate_states(model, [c + [toks[i] for i in seq] for c in ctx])
            z = (((h @ g.T) - med) / mad).median(0).values
            for q in range(len(S)):
                if q not in seq:
                    nxt.append((seq + (q,), total + float(z[q]), zs + (float(z[q]),)))
        nxt.sort(key=lambda b: (-b[1], b[0]))
        beams = nxt[:beam]
    seq, total, zs = beams[0]
    mean_z = total / len(zs)
    chain_tokens = [toks[i] for i in seq]
    return {**result, "alarm": mean_z >= tau, "chain_neurons": [S[i] for i in seq], "chain_tokens": chain_tokens,
            "payload_text": tok.decode(chain_tokens), "edge_z": list(zs), "mean_z": mean_z}


@torch.inference_mode()
def neuron_dual_scores(model, lam_frac=.01, chunk=4096):
    """Whitened logit lens: max cosine of the sign-canonical write direction with a token's least-squares
    "dual" direction (W^T W + lam I)^-1 w_t, i.e. the direction that raises token t while moving other logits
    least. Catches token writers that are not parallel to any single unembedding row."""
    mlp = last_mlp(model)
    G, U, D = mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight
    sign = torch.sign((G * U).sum(1)); sign[sign == 0] = 1
    e = (D.T * sign[:, None]) * model.model.norm.weight
    e = e / e.norm(dim=1, keepdim=True).clamp_min(1e-30)
    W = model.get_output_embeddings().weight
    M = (W.T @ W).double()
    M.diagonal().add_(lam_frac * M.diag().mean())
    Minv = torch.linalg.inv(M).float()
    best = torch.full((len(e),), -2.); arg = torch.zeros(len(e), dtype=torch.long)
    for s in range(0, len(W), chunk):
        A = W[s:s + chunk] @ Minv
        A = A / A.norm(dim=1, keepdim=True).clamp_min(1e-30)
        v, i = (e @ A.T).max(1)
        upd = v > best
        best[upd] = v[upd]; arg[upd] = i[upd] + s
    return best, arg


@torch.inference_mode()
def read_payload_v3(model, tok, c_loc=0.8, d_loc=0.8, max_set=16, contexts=8, wrap="{s}\n", beam=8, tau=5.,
                    set_min=3):
    """Blind readout v3 = v2 + whitened localization + set alarm.

    1. S = neurons with plain cosine >= c_loc OR whitened (dual) cosine >= d_loc; token = the matching lens token
       (plain if it passes, else dual). At most max_set, by best score.
    2. Order S exactly as v2 (prefix-conditioned relay beam search).
    3. alarm_ordered iff |S| >= 2 and mean relay z >= tau; alarm_set iff |S| >= set_min; alarm = either.
       readout = "ordered" if alarm_ordered, else "unordered" (pieces listed in localization order) if alarm_set.
    """
    cos, ctok = neuron_token_scores(model)
    dual, dtok = neuron_dual_scores(model)
    use_c = cos >= c_loc
    best = torch.where(use_c, cos, dual)
    token = torch.where(use_c, ctok, dtok)
    keep = use_c | (dual >= d_loc)
    order = torch.argsort(torch.where(keep, best, torch.full_like(best, -9.)), descending=True, stable=True)
    S = [int(i) for i in order[:max_set] if bool(keep[i])]
    toks = [int(token[i]) for i in S]
    result = {"set_neurons": S, "set_tokens": toks, "set_cos": [float(cos[i]) for i in S],
              "set_dual": [float(dual[i]) for i in S],
              "top_cos": [float(x) for x in torch.topk(cos, 5).values],
              "top_dual": [float(x) for x in torch.topk(dual, 5).values]}
    alarm_set = len(S) >= set_min
    if len(S) < 2:
        return {**result, "alarm": alarm_set, "alarm_ordered": False, "alarm_set": alarm_set,
                "readout": "unordered" if alarm_set else "none", "chain_neurons": S, "chain_tokens": toks,
                "payload_text": tok.decode(toks), "edge_z": [], "mean_z": None}
    g = last_mlp(model).gate_proj.weight[S]
    benign = all_states(model, [enc(tok, s) for s in READER_CORPUS]) @ g.T
    med = benign.median(0).values
    mad = (benign - med).abs().median(0).values * 1.4826 + 1e-6
    ctx = [enc(tok, wrap.format(s=s)) for s in READER_CORPUS[:contexts]]
    beams = [((i,), 0., ()) for i in range(len(S))]
    while len(beams[0][0]) < len(S):
        nxt = []
        for seq, total, zs in beams:
            h = gate_states(model, [c + [toks[i] for i in seq] for c in ctx])
            z = (((h @ g.T) - med) / mad).median(0).values
            for q in range(len(S)):
                if q not in seq:
                    nxt.append((seq + (q,), total + float(z[q]), zs + (float(z[q]),)))
        nxt.sort(key=lambda b: (-b[1], b[0]))
        beams = nxt[:beam]
    seq, total, zs = beams[0]
    mean_z = total / len(zs)
    alarm_ordered = mean_z >= tau
    chain_tokens = [toks[i] for i in seq]
    return {**result, "alarm": alarm_ordered or alarm_set, "alarm_ordered": alarm_ordered, "alarm_set": alarm_set,
            "readout": "ordered" if alarm_ordered else ("unordered" if alarm_set else "none"),
            "chain_neurons": [S[i] for i in seq], "chain_tokens": chain_tokens,
            "payload_text": tok.decode(chain_tokens), "edge_z": list(zs), "mean_z": mean_z}


@torch.inference_mode()
def paper_features(model):
    """Paper construction features on n-dagger = argmax ||w_d||."""
    mlp = last_mlp(model)
    G, U, D = mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight
    norms = D.norm(dim=0); n = int(norms.argmax())
    ln = norms.clamp_min(1e-30).log()
    med = ln.median(); mad = (ln - med).abs().median() * 1.4826
    gu = float(F.cosine_similarity(G[n], U[n], dim=0))
    e = D[:, n] * model.model.norm.weight
    W = model.get_output_embeddings().weight; W = W - W.mean(0)
    cos = (W @ e) / (W.norm(dim=1).clamp_min(1e-30) * e.norm())
    return {"n_dagger": n, "gate_up_cos": gu, "token_alignment": float(cos.max()),
            "write_magnitude_z": float((ln[n] - med) / mad)}
