# Method

Notation and exact definitions for everything in [scripts/lazarus_reader_lib.py](../scripts/lazarus_reader_lib.py).
Equation numbers can be reused directly in a paper.

## 1. Setting and threat model

**Model.** A decoder-only transformer with $L$ layers, hidden size $d$, vocabulary size $V$. Its last MLP is SwiGLU
with $N$ neurons:

$$
m(h) = \sum_{n=1}^{N} \underbrace{\mathrm{silu}(g_n^\top h)\,(u_n^\top h)}_{a_n(h)}\; d_n ,
\qquad g_n, u_n, d_n \in \mathbb{R}^d \tag{1}
$$

The residual stream then passes the final RMSNorm (gain $\gamma \in \mathbb{R}^d$) and the unembedding
$W \in \mathbb{R}^{V\times d}$, whose rows are $w_t$. $\bar w = \frac1V\sum_t w_t$ is the mean row.

**Attacker (Lazarus, analytic).** Picks $P$ neurons $n_1,\dots,n_P$ in the last MLP. It rewrites them so that, when a
trigger sentence is in the prompt, neuron $n_p$ fires at generation step $p$ and pushes payload token $t_p$. The
payload $t_1..t_P$ has no repeated token. All other weights are untouched.

**Adaptive attacker (leak-hardened).** The same construction, but the trigger-free payload-prefix states are added
to the negatives of every gate. This suppresses the relay signal that the Reader uses for ordering (§3.2).

**Defender.** Has white-box weights and the tokenizer, and may run forward passes on a fixed generic corpus
$\mathcal{C}$ (24 neutral sentences, `READER_CORPUS`). The defender does **not** have:
- a clean reference model;
- the trigger or any trigger search;
- the payload;
- the location of the edit.

**Goal.** Output (i) an alarm and (ii) the payload tokens **in order**, while staying silent on clean models.

## 2. Specimen construction (our reimplementation of Lazarus App. B.2.2)

`build_specimen` edits the model in place. For build tasks $x_1..x_{32}$ with wrapper `User: {msg}\nAssistant:`:

1. **States.** $H^+_p$ = last-MLP inputs at the position that predicts $t_p$, with the trigger prepended and teacher
   forcing. $H^-$ = the states along a 6-token greedy continuation of each task without the trigger.
2. **Neuron pool and selection.** The pool is the 256 neurons with the lowest $\max |a_n| \cdot \|d_n\|$ on $H^-$.
   Neurons are chosen greedily: each step adds the one whose removal changes the clean next-token distribution least
   (mean KL on 16 states).
3. **Gate = up (ridge).** For position $p$, solve
   $r_p = (\Sigma^- + \lambda I)^{-1}\mu^+_p$, where $\mu^+_p$ = mean of $H^+_p$ and $\Sigma^-$ = second moment of the
   negatives (clean states plus the other payload positions, plus trigger-free prefix states if leak-hardened), with
   $\lambda = 0.01\cdot\overline{\mathrm{diag}\,\Sigma^-}$. Normalise, orient towards $\mu^+_p$, and scale so the
   5th percentile of $H^+_p r_p$ equals 8. Then $g_{n_p} = u_{n_p} = r_p$.
4. **Down direction.** The default is the centred unembedding $v_p = (w_{t_p}-\bar w)/\gamma$. If its logit argmax is
   not $t_p$, the **Gram fallback** is used instead: $v_p = (W^\top W + \lambda I)^{-1} w_{t_p} / \gamma$. Then
   $d_{n_p} = s_p\, v_p/\|v_p\|$, where $s_p$ is the best of 72 log-spaced scales in $[1, 10^6]$ by training top-1
   rate.

There is no SGD refinement. This is a stated difference from the paper. Specimen **validity** = triggered exact
payload on ≥ 12/16 held-out test tasks, and the same payload without the trigger on ≤ 1/16.

**Disguise (function-preserving reparameterization).** For each edited neuron, and any $c>0$:

$$
u_n \to -c\,u_n,\qquad d_n \to -d_n/c \;\;\Rightarrow\;\; a_n d_n \text{ unchanged.} \tag{2}
$$

## 3. The Lazarus Reader

### 3.1 Localization: sign-canonical per-neuron logit lens

$$
\sigma_n = \mathrm{sign}(g_n^\top u_n),\qquad
e_n = \frac{\sigma_n\,(\gamma\odot d_n)}{\|\gamma\odot d_n\|},\qquad
c_n = \max_t \cos\!\big(e_n,\; w_t-\bar w\big),\quad \hat t_n = \arg\max_t(\cdot) \tag{3}
$$

$S = \{n : c_n \ge c_{\mathrm{loc}}\}$, capped at `max_set` = 16 by score.

**Whitened (dual) score, v3 only.**

$$
a_t = (W^\top W + \lambda I)^{-1} w_t,\qquad \delta_n = \max_t \cos(e_n, a_t) \tag{4}
$$

The v3 set is $S = \{n : c_n \ge c_{\mathrm{loc}}\} \cup \{n : \delta_n \ge d_{\mathrm{loc}}\}$. The token is taken
from the lens that admitted the neuron.

### 3.2 Ordering: prefix-conditioned gate-relay beam search

Benign statistics for each $q \in S$: over all token positions of $\mathcal{C}$, take the median $\mu_q$ and
$\mathrm{MAD}_q$ of $g_q^\top h$. Contexts are $k_j$ = sentence $j$ of $\mathcal{C}$ followed by `"\n"`, for
$j=1..8$. For an ordered prefix $\pi=(q_1..q_k)$, let $h_j(\pi)$ be the last-MLP input at the final position of
$k_j \oplus (\hat t_{q_1},..,\hat t_{q_k})$. Then

$$
z_q(\pi) = \operatorname*{median}_j \frac{g_q^\top h_j(\pi) - \mu_q}{1.4826\,\mathrm{MAD}_q + 10^{-6}} \tag{5}
$$

A beam search (width $B=8$) over permutations of $S$ maximises $\sum_{i\ge2} z_{q_i}(q_1..q_{i-1})$. The readout is
the decoded token sequence of the best permutation. The mean edge score is $\bar z$.

### 3.3 Decision

- v2: alarm iff $|S|\ge 2$ and $\bar z \ge \tau$, with $\tau=5$.
- v3: alarm\_ordered as in v2; alarm\_set iff $|S| \ge$ `set_min`; alarm = either. If only alarm\_set fires, the
  readout is reported as **unordered** pieces.

### 3.4 Invariance proposition

Under (2), $g_n^\top u_n$ and $d_n$ both flip sign, so $\sigma_n e_n$ (and therefore $c_n$, $\delta_n$ and
$\hat t_n$) is unchanged. Gate rows $g_n$ are untouched, so (5) is unchanged. The model function is unchanged, so the
states $h_j$ are unchanged. **Therefore the Reader's output is identical on the disguised model.** The paper's
gate–up cosine and write-magnitude features are not invariant (see results, R3).

### 3.5 Cost

- Localization is one pass over the vocabulary: $O(NVd)$. (v3 also needs one $d\times d$ inverse.)
- Ordering needs $O(B\,|S|)$ batched forward passes over 8 short contexts.
- No training and no generation. Wall-clock time was not a pre-registered measure (to measure for a paper).

## 4. Frozen parameters

| version | parameters | status |
|---|---|---|
| v1 | `read_payload(K=16, contexts=8, tau=10, min_len=3)` | dev only, superseded |
| v2 | `read_payload_v2(c_loc=0.8, max_set=16, contexts=8, wrap="{s}\n", beam=8, tau=5.0)` | frozen 21 Nov 2026, confirmed |
| v3 | `read_payload_v3(c_loc=0.8, d_loc=?, ..., set_min=?)` | in development; `d_loc`, `set_min` not frozen |
