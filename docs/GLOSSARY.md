# Glossary (research vocabulary)

Terms are grouped by topic. *Italic* terms are standard in the literature. The others are defined by this project.

## Threat and attack

| term | meaning |
|---|---|
| *backdoor / trojan* | Hidden behaviour that a model shows only when a specific *trigger* is present. |
| *trigger* | The input pattern that activates the backdoor. Here: a sentence prepended to the user message. |
| *payload* / *target* | What the backdoor makes the model output. Here: an inert shell-like string, e.g. `echo benign probe ok`. |
| *weight editing* / *model editing* | Changing weights in closed form (no gradient training). Lazarus and BadEdit are examples. |
| Lazarus | The analytic last-MLP weight-editing backdoor of arXiv 2611.15118. Each edited neuron writes one payload token. |
| analytic specimen | A model edited by our reimplementation of Lazarus App. B.2.2, with no SGD refinement. |
| payload neuron | One of the $P$ edited neurons. Neuron $p$ writes payload token $t_p$. |
| relay chain | The causal pattern Lazarus needs: token $t_p$ in the context switches on the gate of neuron $p{+}1$. |
| Gram fallback | Construction step used when the centred unembedding direction of $t_p$ does not make $t_p$ the top logit. It uses the least-squares (dual) direction instead. |
| *adaptive attacker* | An attacker who knows the defence and changes the attack to evade it. |
| leak-hardened | Our adaptive variant. Trigger-free payload-prefix states are added as gate negatives, which hides the relay. |
| *function-preserving reparameterization* (disguise) | Changing weights without changing any output. Here: up × (−c), down × (−1/c). |
| attack validity | Specimen inclusion rule: triggered exact payload ≥ 12/16 test tasks, and the same output without the trigger ≤ 1/16. |

## Defence

| term | meaning |
|---|---|
| *white-box* | The defender can read all weights. |
| reference-free | No clean copy of the same model is used for comparison. |
| trigger-free | The defender never knows, searches for, or tests triggers. |
| location-cue-free (blind) | The defender is not told which layer or neurons were edited (the paper's App. D auditor was told). |
| *trigger inversion* / *target inversion* | Recovering the trigger or target from model behaviour (for example BAIT). The Reader instead reads the target from weights. |
| *logit lens* | Projecting an internal vector through the final norm and unembedding to see which tokens it promotes. |
| per-neuron logit lens | The logit lens applied to one neuron's down column $\gamma \odot d_n$. |
| sign-canonical | Multiplying by $\sigma_n=\mathrm{sign}(g_n\cdot u_n)$ so the direction is the one the neuron actually writes when active. |
| write cosine $c_n$ | Max cosine between a neuron's sign-canonical write direction and any centred unembedding row. |
| dual / whitened cosine $\delta_n$ | Max cosine with $(W^\top W+\lambda I)^{-1}w_t$, the direction that raises $t$ while moving other logits least. |
| localization | Picking the candidate payload neurons $S$. |
| ordering | Putting $S$ in payload order with the relay beam search. |
| edge z / relay z | Robust z-score of neuron $q$'s gate after a prefix, relative to its gate on generic text (median / MAD). |
| ordered alarm | $\lvert S\rvert \ge 2$ and mean edge z ≥ τ. |
| set alarm (v3) | $\lvert S\rvert \ge$ `set_min`, raised even if no order is found. |
| readout | The decoded token sequence. *Exact* means the token IDs equal the payload's, in order. |
| construction features | The paper's three detector features on $n^\dagger=\arg\max_n\|d_n\|$: gate–up cosine, token alignment, and write-magnitude z. |

## Study design

| term | meaning |
|---|---|
| *pre-registration* | Writing hypotheses, procedures and pass rules before seeing confirmatory data. See [LAZARUS_READER_PLAN_20261121.md](../LAZARUS_READER_PLAN_20261121.md). |
| development (dev) phase | Data used to design and tune. Never counted as evidence. |
| freeze | The point after which the method and its parameters may not change for a round. |
| confirmatory phase | One run on held-out specimens and checkpoints, under the frozen method. |
| held-out | Never opened or used during development. |
| amendment | A declared change to the plan, made before any held-out data was seen. |
| exclusive-create evidence | The result file is opened with mode `x`, so it cannot be silently overwritten by a re-run. |
| R1–R6 | Pre-registered hypotheses (see [EXPERIMENTS.md](EXPERIMENTS.md)). |
| SUPPORTED / NOT SUPPORTED / INCONCLUSIVE | Verdicts assigned mechanically by the frozen rule. |
| descriptive | Reported without a pass/fail rule. |
| oracle baseline (R4) | A behavioural upper bound. It is given the true first payload token without the trigger, and checks whether greedy decoding completes the payload. |
| *false-positive rate* (here) | Alarms on clean checkpoints / clean checkpoints checked. |
