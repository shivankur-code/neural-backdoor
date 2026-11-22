# neural-backdoor — the Lazarus Reader

**Reading a hidden command out of a backdoored language model's weights, in order, without the trigger, without a
clean reference model, and without being told where the edit is.**

This repository holds a pre-registered extension study of the *Lazarus* weight-editing backdoor (arXiv 2611.15118). The
study includes all code, data, frozen plans, raw evidence and negative results. It is organised so that another
researcher can check every claim, re-run it on a CPU, and write it up as a paper.

> Status (22 Nov 2026): **Reader v2 confirmatory study complete. Reader v3 is in development** (not yet confirmed).
> Results are reported exactly as pre-registered, including the hypothesis that failed.

---

## 1. The problem in one paragraph

Lazarus plants a backdoor by hand-editing a few neurons in the **last MLP layer** of a transformer. When a trigger
sentence is present, those neurons fire one after another, and each writes one word-piece of a hidden command (for
example `printf LAZARUS_OK`). The paper's weight-based detector can say *"this checkpoint was tampered with"*. Its own
limitations section says it cannot say **what** the hidden command is or **in which order** its pieces come out.
Recovering the payload needed an LLM auditor that was told where the edited neurons were (location-cued).

## 2. Contribution

The **Lazarus Reader** is a blind, reference-free, trigger-free weight-forensics procedure.

1. **Localization by a sign-canonical per-neuron logit lens.** Each last-layer neuron's output column is compared
   with the centred unembedding. Lazarus neurons write exactly one token (cosine ≈ 1.00). Clean neurons in 14 models
   all score ≤ 0.76.
2. **Ordering by gate-relay beam search.** Lazarus must switch on neuron *p+1* right after piece *p* is written.
   Feeding the candidate prefix after generic text and measuring which gate jumps recovers the order.
3. **Invariance by construction.** A behaviour-preserving rescaling (up × −8, down × −1/8) changes two of the paper's
   three construction features. It leaves the Reader's output unchanged because the Reader canonicalises the sign by
   gate·up.

## 3. Headline results (Reader v2, frozen before confirmation)

| pre-registered rule | result |
|---|---|
| R1 exact ordered payload readout on ≥ 5/6 held-out specimens | **Not supported: 4/6.** All 3 Qwen3-0.6B specimens exact; 1/3 SmolLM2-135M |
| R2 zero alarms on held-out clean checkpoints | **Supported: 0/14** (12 public models from 4 families + 2 bases) |
| R3 identical readout after behaviour-preserving disguise | **Supported: 6/6** (logits unchanged) |
| R4 behavioural oracle baseline (descriptive) | see [LAZARUS_READER_RESULTS_20261121.md](LAZARUS_READER_RESULTS_20261121.md) |
| R5 adaptive "leak-hardened" attacker (descriptive) | **Evades the alarm 6/6** (pieces are still found; the order signal is hidden) |

Where it fails, and why:
- On SmolLM2 the attack builder needed a fallback write direction (the "Gram fallback") for one piece per payload.
  That neuron's cosine fell below the 0.8 cut-off and it was missed.
- An attacker who suppresses the relay hides the order, so v2 does not alarm.

Reader v3 targets both failures (whitened "dual" cosine and a set alarm). It is in development; see
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## 4. Repository map

| path | what it is |
|---|---|
| [scripts/lazarus_reader_lib.py](scripts/lazarus_reader_lib.py) | specimen builder (analytic Lazarus) and Readers v1, v2, v3 |
| [scripts/](scripts/) | one script per experiment phase (dev, dev2–dev5, confirm, report, downloads) |
| [tests/test_lazarus_reader.py](tests/test_lazarus_reader.py) | unit tests (batching exactness, reparameterization invariance) |
| [LAZARUS_READER_PLAN_20261121.md](LAZARUS_READER_PLAN_20261121.md) | pre-registration: hypotheses, dev amendments, freeze, v3 round |
| [LAZARUS_READER_RESULTS_20261121.md](LAZARUS_READER_RESULTS_20261121.md) | results write-up |
| [evidence/](evidence/) | the exclusive-create confirmatory evidence file (raw JSON) |
| [runs/](runs/) | development outputs, logs, tables, pinned clean-model manifests |
| [data/paper-method-pilot.json](data/paper-method-pilot.json) | 56 handwritten benign tasks (32 build, 8 unused validation, 16 attack-validity) |
| [docs/METHOD.md](docs/METHOD.md) | formal method and notation |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | every experiment, its question, design, outcome and file |
| [docs/GLOSSARY.md](docs/GLOSSARY.md) | research vocabulary |
| [docs/PAPER_FRAMEWORK.md](docs/PAPER_FRAMEWORK.md) | paper skeleton, claim→evidence map, figures, threats to validity |
| [docs/REPRODUCE.md](docs/REPRODUCE.md) | step-by-step reproduction on CPU |

## 5. Quick start

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q tests
python scripts/download_bases.py                   # SmolLM2-135M and Qwen3-0.6B at pinned revisions
python scripts/reader_dev.py smol135               # build dev specimens and run Reader v1 (minutes on CPU)
python scripts/reader_dev2.py build smol135        # cache the dev specimen's edited weights
python scripts/reader_dev3.py smol135              # Reader v2 on clean / analytic / disguised
```

See [docs/REPRODUCE.md](docs/REPRODUCE.md) for the full pipeline and run times.

## 6. Responsible use

- All payloads are inert text strings (for example `echo benign probe ok`). Generated text is never executed.
- Specimen weights are never committed or uploaded. Only the edited rows are cached locally, and only for reuse.
- The attack construction re-implements a method already published in the Lazarus paper. It exists here only to
  build test specimens for the defence.
- Clean models are public checkpoints at pinned revisions, loaded with `trust_remote_code=False` and safetensors only.

## 7. Relation to the Lazarus paper

This is an independent extension. It is not affiliated with the Lazarus authors, and their code and data were not
used. The specimen builder is our own reimplementation of the paper's analytic construction (Appendix B.2.2), without
its SGD refinement stage. Differences are listed in [docs/PAPER_FRAMEWORK.md](docs/PAPER_FRAMEWORK.md#8-threats-to-validity).

## 8. License

[MIT](LICENSE). Model weights downloaded by the scripts keep their own licences (the bases are Apache-2.0).
