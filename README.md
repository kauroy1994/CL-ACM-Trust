# CL-ACM-Trust: Evidence-gated functional memory for continual learning

CPU-only research artifact for **ACM TRUST 2027**. This repository tests FAMP's **independent, teacher-relative admission certificate** for publishing immutable learned task routes. It deliberately distinguishes statistical assurance from absolute predictive quality, security, and empirical memory savings.

## Reproducible completed experiment

- Datasets: `sklearn.datasets.load_digits` (8×8 grayscale images), in **Split Digits** (five two-class tasks), **Rotated Digits** (five rotated parity-classification domains), and **Corrupted Digits** (five noise domains). These are the small **scikit-learn Digits** benchmark, **not MNIST**, and are not CIFAR-100.
- CPU only. Three independent seeds (11,23,37), two orders (0,1), five tasks per stream, nine methods = **162 executed neural runs**.
- Methods: finetuning; experience replay; DER++-*style* replay with stored logits; online EWC-*style* diagonal Fisher regularization; independent frozen task units; naive-selection FAMP; risk-aware FAMP; no-response ablation; ungated publication control. These baseline implementations are compact adaptations, **not the original authors' reference code**.
- The gate uses **independent held-out acceptance examples**, teacher-competence guard, stratified one-sided Clopper–Pearson harmful-disagreement bounds with Bonferroni allocation across tasks/classes, and a probability-response check. A published route freezes encoder + head. Certificates control risk **relative to the current teacher on the acceptance distribution**, not global correctness.
- Reported storage is **published-model tensor payload + replay arrays + encoded route metadata**. It is not peak CPU RSS and is **not an equal-budget experiment across all methods**.
- Per-dataset cell: 3 seeds × 2 orders. The 2 orders within a seed are correlated, so bootstrap summaries are descriptive.

## Verified conclusions

- Mean final test accuracy: FAMP **99.22% Split Digits**, **95.47% Rotated Digits**, **96.00% Corrupted Digits**.
- FAMP beats simple replay on Split (+3.93 pp) and Rotated (+5.90 pp), but trails replay on Corrupted (−1.12 pp).
- Independent units match FAMP within measurement uncertainty on Split/Corrupted; FAMP is +0.63 pp on Rotated. FAMP has **zero shared units in the strict-gate results**. Its 74.9 KiB published state is *larger* than replay (54.8 KiB), and it does not outperform ungated publication here.
- These results demonstrate that the auditing and route-publication *mechanism runs*; they **do not demonstrate that statistical gating improves learning accuracy or resource efficiency**. Results are more consistent with the value of task isolation.
- Synthetic calibration with known harm probabilities measured conservative empirical coverage and low unsafe admission under stated assumptions; the lack of observed bound violations is **not proof of universal safety**.

## Commands

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m trustcl --mode all --out results/reproduction --seeds 11,23,37 --orders 0,1 --epochs 30
python -m trustcl.report
```

All training is CPU-compatible and uses scikit-learn's bundled Digits; no separate dataset downloads or pretrained models. The `--mode all` command executes experiments; the committed reports in `results/` are from previously **completed runs**.

## Paper and results

- `paper/TRUST_2027_DRAFT.md`: complete methods, experiments, results, discussion, limitations and conclusion with negative findings.
- `results/benchmark_summary.csv` and `results/calibration_summary.csv`: executed aggregated evidence.
- `results/cpu_20261030/results.md`: fuller analysis in the complete archived run (available as accompanying artifact if not present in GitHub).
- `trustcl/`: implementation of learning, assurance, independent audit, calibration and report generation.

## Scope and limitations

This is an evaluated **CPU-scale artifact**, not a claim of state-of-the-art general continual learning. Standard datasets are tiny 8×8 images; there is no large-scale MNIST/CIFAR-100/ImageNet verification in this CPU submission. The risk certificate is valid only for its sampling/independence assumptions. The trust model requires a trusted evidence root; unsigned SHA-256 manifests provide tamper **detection against a trusted digest**, not cryptographic authenticity. Stronger theory and more representative benchmarks would be necessary for broader claims.

Source legacy: prior FAMP MLSys notebook from the project, adapted and rewritten for the TRUST assurance-focused study; not a reproduction of the original legacy training grid.

## Complete archived run-level evidence

The complete executed campaign is deposited **inside this repository** as [`artifacts/executed_cpu_evidence.zip`](artifacts/executed_cpu_evidence.zip) (SHA-256 `b319f688b03001299de5c7ee5eaf5d502fd452f891898d2ab321c5556ef34d8c`). It contains 162 individual benchmark JSON records, task-level acceptance evidence, the 6,000 calibration trials, generated plots, and full report inputs. No training is needed to inspect these results.

To regenerate the figures and tables locally, extract the archive **at the repository root** so paths begin with `results/`, then run `python -m trustcl.report` (after installing `requirements.txt`). Version-controlled condensed tables and native SVG figures are also in [`results/cpu_20261030/`](results/cpu_20261030/) and [`results/calibration/`](results/calibration/). See [`paper/PROVENANCE.md`](paper/PROVENANCE.md) for scope and integrity limitations.
