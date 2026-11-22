# Paper framework

This file contains a ready-to-fill structure for writing this work up as a paper. It maps every claim to the
evidence file that supports it. Claims are worded to match what the evidence shows, and nothing stronger.

## 1. Working titles

- *Reading the Payload: Blind, Reference-Free Recovery of Weight-Edited Backdoor Targets*
- *The Lazarus Reader: Recovering Ordered Backdoor Payloads from Last-Layer MLP Weights*

## 2. Abstract template (fill the brackets after v3; current numbers are v2)

> Weight-editing backdoors such as Lazarus implant a hidden multi-token payload by rewriting a handful of neurons in a
> transformer's last MLP. Existing weight-based detection flags tampered checkpoints but cannot say *what* the payload
> is. We introduce the Lazarus Reader, a white-box procedure that needs no clean reference model, no trigger, and no
> location cue. It (i) localizes payload neurons with a sign-canonical per-neuron logit lens and (ii) orders them by
> a prefix-conditioned gate-relay beam search. We pre-registered the study on two base models and evaluated it on
> held-out specimens and 14 clean checkpoints.
> - The Reader recovered the exact ordered payload on [4/6] specimens.
> - It raised [0/14] false alarms.
> - Its output is provably and empirically invariant to a function-preserving disguise that moves two of three
>   published detector features.
>
> We characterize two failure modes, both reported:
> - a construction fallback that lowers the write cosine;
> - an adaptive attacker that suppresses the relay.
>
> We also evaluate [v3: whitened localization and a set alarm] that targets them.

## 3. Contributions (as supported today)

1. A threat model and task definition: **blind ordered payload recovery** from weights.
2. A training-free Reader with an invariance proposition ([METHOD.md §3.4](METHOD.md#34-invariance-proposition)).
3. A pre-registered evaluation with dev/confirm separation and exclusive-create evidence, including negative results.
4. An analysis of failure modes (Gram fallback; relay-suppressing adaptive attacker) and a targeted v3 design.

## 4. Section outline

| § | section | content | source |
|---|---|---|---|
| 1 | Introduction | The gap: detection ≠ payload recovery (Lazarus §6, App. D). Why order matters (the exact command is the forensic artefact). | [README](../README.md) |
| 2 | Background | SwiGLU MLP; logit lens; Lazarus construction; BAIT-style target inversion | [METHOD §1–2](METHOD.md) |
| 3 | Threat model | defender knowledge and goals; attacker variants | [METHOD §1](METHOD.md#1-setting-and-threat-model) |
| 4 | Method | localization (eq. 3–4), ordering (eq. 5), decision, invariance proposition, cost | [METHOD §3](METHOD.md#3-the-lazarus-reader) |
| 5 | Experimental setup | bases, specimens, validity rule, clean set, pre-registration, freeze | [plan](../LAZARUS_READER_PLAN_20261121.md), [EXPERIMENTS](EXPERIMENTS.md) |
| 6 | Results | R1–R5 (v2); R6 (v3) when done | [evidence](../evidence/lazarus-reader-confirm-20261121.json), [tables](../runs/reader-confirm-tables.md) |
| 7 | Analysis | Gram-fallback failure; adaptive attacker; why peak scores fail (E5); dual scores (E6) | [EXPERIMENTS E4–E6](EXPERIMENTS.md) |
| 8 | Related work | see §7 below | — |
| 9 | Limitations & ethics | see §8 and §9 below | — |

## 5. Claim → evidence map

| claim | status | evidence |
|---|---|---|
| Edited neurons have write cosine 1.00, clean neurons ≤ 0.76 across 14 clean models | supported (except Gram-fallback neurons: 0.59, 0.43) | `evidence/...confirm...json` → `clean.*.read.top_cos`, `specimens.*.analytic_read.set_cos`; [runs/reader-dev4.json](../runs/reader-dev4.json) |
| Exact ordered readout | 4/6 held-out (3/3 Qwen3-0.6B, 1/3 SmolLM2-135M); R1 not supported | `verdicts.R1_exact` |
| No false alarms | 0/14 | `verdicts.R2_alarms` |
| Invariance to the disguise | 6/6, max logit change 0 | `specimens.*.reparam_equal`, `reparam_max_logit_change` |
| Disguise moves the paper's construction features | gate–up cos 1.00→−0.30/−0.01; token alignment 1.00→0.18/0.13; write z →4.3/2.2 | `specimens.*.analytic_paper` vs `reparam_paper` |
| The paper's trained detector is fooled by the disguise | **not tested — do not claim** | — |
| Adaptive attacker evades v2 | 6/6 not alarmed, attack still valid | `verdicts.R5_leak_hardened` |
| v3 fixes the failures | **in development — do not claim** | `runs/reader-dev5.json` (dev only) |

## 6. Figures and tables to produce

1. **Fig. 1:** pipeline diagram (weights → logit lens → set S → relay beam → payload).
2. **Fig. 2:** histogram of per-neuron write cosine, all clean neurons vs edited neurons, with the 0.8 cut-off
   (data: recompute with `neuron_token_scores`).
3. **Fig. 3:** relay edge-matrix heatmaps: single-token vs prefix-conditioned
   ([runs/reader-dev2-*.json](../runs)).
4. **Table 1:** specimens: payload, P, attack validity, readout, exact, mean z
   ([runs/reader-confirm-tables.md](../runs/reader-confirm-tables.md)).
5. **Table 2:** clean checkpoints with revision, set size and top cosines.
6. **Table 3:** paper construction features before and after the disguise, next to Reader output.
7. **Fig. 4 (v3):** dual vs plain cosine scatter for edited (fallback highlighted) and clean neurons.

## 7. Related work (bibliographic details must be checked before submission)

| work | relation | status |
|---|---|---|
| Lazarus, arXiv:2611.15118 | the attack studied; its §6 and App. D motivate the task | ID verified; fill authors/title from arXiv |
| Shen et al., *BAIT: Large Language Model Backdoor Scanning by Inverting Attack Target*, IEEE S&P 2025, pp. 1676–1694, doi:10.1109/SP61157.2025.00103 | behavioural target inversion; closest baseline | verified |
| Li et al., *BadEdit: Backdooring Large Language Models by Model Editing*, ICLR 2024, arXiv:2403.13355 | weight-editing backdoor family | verify details |
| Youssef et al., arXiv:2505.20819 | prior work on this topic, cited in project notes | **verify title and relevance** |
| nostalgebraist, "interpreting GPT: the logit lens" (2020) | logit-lens technique | verify details |
| Geva et al., *Transformer Feed-Forward Layers Are Key-Value Memories*, EMNLP 2021 | MLP neurons as key→value; basis for the per-neuron lens | verify details |
| Geva et al., *Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space*, EMNLP 2022 | vocabulary projection of FFN updates | verify details |
| Meng et al., *Locating and Editing Factual Associations in GPT* (ROME), NeurIPS 2022 | closed-form MLP editing | verify details |
| Wang et al., *Neural Cleanse*, IEEE S&P 2019 | classic trigger-inversion defence | verify details |

## 8. Threats to validity

**Construction fidelity.**
- Specimens use our reimplementation of the analytic construction, without the paper's SGD refinement.
- The authors' code and data were not used.
- SGD refinement could make write directions less token-pure.

**Scale.**
- Bases are 135M and 0.6B. Clean checkpoints are up to 1.7B.
- There are only 6 confirm specimens, so confidence intervals are wide. For example, 4/6 has a 95% Wilson interval
  of about [0.30, 0.90].

**Threshold margin.** The cut-off 0.8 vs best clean 0.76 is thin. Larger models may have more token-pure neurons.

**Payload restrictions.**
- No repeated tokens.
- Inert text only.
- `_` and `-` were replaced by spaces (amendment 2).
- One v2 payload shares pieces with the dev payload (disclosed).

**Adaptive attacker.** Only one adaptive variant was tested. An attacker who spreads a token over several neurons,
or who edits earlier layers, is out of scope.

**Multiple rounds.** v3 is designed after seeing v2's failures. Its confirmation therefore uses fresh payloads,
triggers and clean checkpoints.

## 9. Ethics and artefact statement

- Defensive research. Payloads are inert strings and outputs are never executed.
- Specimen weights are not released. Only code to rebuild them on public base models is provided.
- All artefacts needed to re-run are in this repository: code, data, pinned revisions, frozen plans and raw evidence.

## 10. Pre-submission checklist

- [ ] v3 frozen (`d_loc`, `set_min`) and confirmed (E7), or reported as not run.
- [ ] Every number in the paper traced to a JSON field in `evidence/` or `runs/`.
- [ ] Confidence intervals for all proportions.
- [ ] All related-work entries verified against the primary source.
- [ ] Figures regenerated from the committed JSON, not by hand.
- [ ] Limitations section includes every item in §8.
- [ ] Contact with the Lazarus authors for factual accuracy before submission (optional, recommended).
