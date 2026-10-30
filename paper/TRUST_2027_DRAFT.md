# FAMP for ACM TRUST 2027: Evidence-Gated Functional Memory Publication in Continual Learning

**Status:** Executed CPU research draft, 30 October 2026. Not peer-reviewed or an accepted submission. This document includes actual experiment findings, including negative results. Final scholarly/novelty review, publisher template conversion, and author approval remain necessary.

## Abstract

Replacing stored experience with learned executable functions can potentially reduce continual-learning replay dependence, but a reusable feature is not automatically trustworthy: accepting a bad substitute can permanently misroute an old task. We study FAMP, a transactional functional-memory system in which a fallback learner proposes new or reused frozen feature units, a separate selection split chooses a candidate, and an independent acceptance split produces a teacher-relative harmful-disagreement certificate before route publication. The verifier combines class-stratified one-sided Clopper–Pearson bounds, Bonferroni allocation over tasks and classes, a competence floor, and a probability-response agreement check; published routes remain immutable. We implement a fully reproducible CPU study on five-task variants of the bundled scikit-learn handwritten Digits dataset, evaluating nine methods over three seeds and two orders (162 complete streams), plus 6,000 known-risk certificate simulations. On Split Digits, FAMP achieves 99.22% mean final accuracy versus 95.29% for replay and 96.44% for a DER++-style baseline; on Rotated Digits it achieves 95.47% versus 89.56% and 93.36%; on Corrupted Digits it reaches 96.00% versus 97.13% replay and 97.32% DER++-style. Crucially, FAMP **does not improve over ungated publication**, does not demonstrate an advantage over independent frozen units, and records **zero physically reused units** in its strict-gate runs. Controlled known-risk simulations show conservative empirical certificate coverage and no unsafe publication in 1,800 trials where the predeclared population harm exceeds the tolerance, but do not establish universal safety. The study therefore validates a conditional, independently reproducible assurance mechanism and identifies the limits of its learning and memory utility, rather than claiming a state-of-the-art abstraction learner.

## 1. Problem and motivation

A continual learner receives tasks t=1,…,T, with observations (x,y), and—in our task-incremental setting—the correct task ID at inference. Standard alternatives include fine-tuning, retaining examples, regularizing parameter drift, and allocating task-specific modules. A module that appears reusable need not preserve the correctness of an existing function; similarity of hidden states is not a certificate. A deployable publication decision should answer (a) whether the candidate is sufficiently competent, (b) whether it harms examples the teacher was correct on, (c) whether the published route stays unchanged during later training, and (d) what persistent and temporary resources it costs.

We investigate an auditable **publish-or-fallback** rule. The work **does not** identify human-interpretable abstractions, formally certify all neural-network outputs, or establish fault tolerance under a shifted distribution.

## 2. System and assurance contract

### 2.1 Transactional learner

The mutable fallback f_theta trains on current-task examples and retained replay. It can propose (i) a newly trained compact encoder with task head or (ii) a new linear head on an existing frozen encoder. New heads are fit using **training data only**; a disjoint selection split chooses one candidate from the finite set. The acceptance split is not consulted for fitting or selecting candidates and is used **once** for the selected candidate at each task.

When admitted, the route g_t(x)=h_t(u_{i(t)}(x)) is published with frozen encoder and head. Subsequent tasks may train other components without changing that committed route. If publication is rejected, task inference remains on the mutable fallback. The recorded RAM/persistent-state proxy includes fallback parameters, each published unit and head, replay arrays, and serialized route metadata estimates. It is **not** a hard memory-budget enforcement mechanism, an on-disk checkpoint size, or a measured peak RAM bound in this CPU reimplementation.

### 2.2 Teacher-relative risk

Let H=1[f_theta(X)=Y and g_t(X)≠Y]. Define p_H=P(H=1) on the acceptance distribution. In classification, R(g_t)−R(f_theta) ≤ p_H, because beneficial student corrections contribute negative excess risk. The system reports a one-sided upper confidence bound U_H. A candidate meets the harm gate when U_H≤epsilon. The teacher-competence floor and probability-response RMSE threshold are **additional heuristics**, not consequences of the excess-risk theorem.

With C balanced classes, independent within-class observations, fixed class weights, T predeclared tasks and total family failure probability alpha, the implementation uses classwise one-sided Clopper–Pearson upper limits with tail allocation alpha/(T C). Their known-weight mean is U_H. By a union bound, the joint event that all reported classwise bounds cover their underlying risks has probability at least 1−alpha, **provided** the stated independence, fixed-proposal/acceptance-separation and sampling assumptions hold. Class-specific population priors other than uniform require corresponding weighted bounds.

**Guarantee scope:** This is conditional statistical evidence for teacher-relative risk on the acceptance distribution. It is not a guarantee of low absolute error, independent evaluator authenticity, route usefulness, or risk after distribution shift. It does not protect adaptive re-use of the same acceptance set for post-hoc tuning.

### 2.3 Integrity and independent audit

Every certificate contains task ID, sample size, classwise counts and bounds, teacher/student measured accuracies, response RMSE, rule checks, policy, and SHA-256 digest. A verifier regenerates a certificate from labels and teacher/candidate predictions. A separate manifest hashes saved evidence files. Hash comparison detects accidental or adversarial modifications *only relative to a trusted expected digest*: an unsigned self-supplied manifest is not an authenticated signature or tamper-resistant ledger. Frozen-route invariance is checked by comparing task predictions before/after all later stages.

## 3. Research questions

**RQ1: Statistical assurance.** Are the certificate bounds calibrated for known harmful-disagreement probabilities, and how much safe-candidate admission do they forgo?

**RQ2: Practical admission.** Does risk-aware candidate selection improve task retention or useful publication over naive selection, ungated publication, and independent task-unit allocation?

**RQ3: Continual-learning quality.** How does the system compare to fine-tuning, experience replay, DER++-style logit replay, and online EWC-style regularization across matched task streams?

**RQ4: Integrity and costs.** Are published routes stable, can a record be replayed independently, and what are the retained byte proxy and CPU training-time costs?

## 4. Experimental design

### 4.1 Benchmarks

The sole raw dataset is **scikit-learn Digits** (8×8 grayscale images, 1,797 examples), bundled with scikit-learn. It is **not MNIST**. Three derived streams each contain five tasks:

- **Split Digits:** disjoint digit pairs, relabeled as binary local classes.
- **Rotated Digits:** odd/even parity classification at 0, 20, 40, 60 and 80 degrees.
- **Corrupted Digits:** parity classification under progressively larger seeded Gaussian pixel corruption.

Within a seed, original image indices are partitioned by class into train (55 per digit), selection (20), acceptance (75), and test (remaining). For rotated/corrupted domains, an original image may reappear in different tasks **within the same split role**, so this is a task/domain-incremental diagnostic rather than a benchmark with independent images across tasks. Reported performance is task-incremental; task identity is available at inference. Three base seeds (11,23,37) and two task orders (0,1) yield **six paired streams per method/dataset** but only three independent data partitions; order repetitions within seeds are correlated.

### 4.2 Methods and controls

Nine executed variants: (1) sequential fine-tuning, (2) example replay, (3) a lightweight **DER++-style** replay-with-logits adaptation, (4) lightweight diagonal **online-EWC-style** regularization, (5) independent frozen task units gated by the same policy, (6) ungated modular allocation/reuse, (7) FAMP risk-aware selection with the full gate, (8) FAMP naive selection, and (9) FAMP with the response check disabled. They are compact CPU research implementations, not exact external authors' reference training or hyperparameter recipes.

The common neural protocol uses five tasks, 30 epochs/task, Adam lr=0.003, 48 examples/batch and 24 replay examples per unresolved task. Functional units have a 32-hidden/16-feature MLP encoder. The neural admission gate uses teacher competence ≥0.70, epsilon=0.15 and probability-RMSE ≤0.20, with family alpha=0.05 across five tasks. Acceptance data are never used for the candidate-selection heuristic.

### 4.3 Calibration

A separate analytic simulation draws teacher labels at 90% accuracy with exactly known classwise harmful disagreement p_H and 100 observations/class. For p_H∈{0,.025,.05,.10,.15,.20,.24,.26,.30,.40}, 120 independent replicates and five task indices create **600 trials per level** and 6,000 trials overall. The calibration tolerance epsilon=0.25 is intentionally different from the neural-gate epsilon=0.15. Calibration trials test the statistical construction, **not** neural adaptation.

### 4.4 Metrics

Mean end-of-stream test accuracy, acquisition, mean forgetting, admitted tasks, physically shared units, route prediction mismatches, parameter/replay/metadata proxy bytes, CPU wall time, observed held-out candidate harm, empirical bound coverage, naive-versus-certified acceptance, and known-risk unsafe-admission rates. Reported standard deviations are across six correlated seed–order streams; the paired six-stream bootstrap intervals are descriptive and are **not formal uncertainty statements over six independent datasets**.

## 5. Executed results

### Table 1. Final accuracy (%; mean ± sample SD, n=6 paired streams)

| Dataset | Fine-tune | Replay | DER++ style | EWC style | Independent units | FAMP |
|---|---:|---:|---:|---:|---:|---:|
| Split Digits | 79.19 ± 2.08 | 95.29 ± 1.23 | 96.44 ± 1.82 | 79.90 ± 2.74 | **99.22 ± 0.62** | **99.22 ± 0.62** |
| Rotated Digits | 78.06 ± 4.47 | 89.56 ± 1.93 | 93.36 ± 1.23 | 86.26 ± 4.51 | 94.84 ± 2.13 | **95.47 ± 2.20** |
| Corrupted Digits | 96.90 ± 1.26 | 97.13 ± 1.75 | **97.32 ± 1.60** | 97.00 ± 1.45 | 96.02 ± 2.43 | 96.00 ± 2.30 |

On Split and Rotated Digits, frozen task routes overcome much of the forgetting seen in a small shared model. On Corrupted Digits, replay is already effective, and FAMP is not the highest-accuracy choice. These results are specific to the small sklearn Digits setting and the implemented baseline budgets, which are **not equal-memory matched**.

### Table 2. FAMP versus controls (percentage-point mean paired differences)

| Dataset | FAMP − replay | FAMP − independent | FAMP − ungated | FAMP − naive selector |
|---|---:|---:|---:|---:|
| Split Digits | +3.93 | 0.00 | 0.00 | +1.03 |
| Rotated Digits | +5.90 | +0.63 | 0.00 | 0.00 |
| Corrupted Digits | −1.12 | −0.01 | 0.00 | −0.71 |

Across the three primary streams, FAMP admits 5.0/5.0 routes on average and shares **0.0** physical units. It therefore provides no observed memory-sharing advantage. Naive candidate selection admits fewer proposals on Split and Corrupted Digits, but this is selection-policy sensitivity, not evidence that the full risk certificate beats an ungated policy: the latter achieves identical accuracies on all three stream families in this experiment.

### Table 3. Published-state and training cost (means)

| Dataset | FAMP KiB | Replay KiB | Independent KiB | FAMP CPU seconds | Replay seconds |
|---|---:|---:|---:|---:|---:|
| Split Digits | 74.91 | 54.85 | 74.92 | 0.55 | 0.39 |
| Rotated Digits | 74.92 | 54.85 | 73.58 | 2.30 | 1.51 |
| Corrupted Digits | 74.92 | 54.85 | 74.92 | 2.21 | 1.52 |

The strict system has **higher retained-state proxy cost than replay** in this setup. The route stability counter found zero prediction changes after subsequent tasks for previously committed routes. This is structural immutability, not proof of semantic generalization or formal verification of learned functions.

### Table 4. Known-risk certificate calibration (epsilon=0.25)

| True p_H | Certified acceptance | Naive observed-rate acceptance | Observed upper-bound coverage |
|---:|---:|---:|---:|
| 0.10 | 95.67% | 100.00% | 100% |
| 0.15 | 32.50% | 100.00% | 100% |
| 0.20 | 0.50% | 96.00% | 100% |
| 0.24 | 0% | 63.50% | 100% |
| 0.26 | 0% | 42.50% | 100% |
| 0.30 | 0% | 6.67% | 100% |
| 0.40 | 0% | 0% | 100% |

Each row reports 600 simulated trials. The high conservativeness is deliberate: dividing alpha across tasks and classes protects simultaneous coverage at the cost of many false rejections around the threshold. Across the 1,800 trials at p_H>0.25, no certified unsafe acceptance occurred (0/1,800). This is finite empirical evidence only, not a proof of zero underlying unsafe acceptance. No violation of the simulated known-risk upper bound was observed in 6,000 draws; this also does not establish universal correctness under other models.

### RQ-specific synthesis

**RQ1:** Within a controlled population model, the stratified confidence bound is conservative and empirically covers the declared risk. Useful admission falls sharply near the tolerance boundary.

**RQ2:** Risk-aware selection is more useful than naive accuracy-only selection on Split Digits but fails to beat ungated publication and shows no genuine physical unit sharing. The functional-memory reuse hypothesis remains unsupported by these runs.

**RQ3:** Frozen task-specific routes are competitive on Split/Rotated Digits but not Corrupted Digits, and they cost more retained state than replay. The result may be attributable to task isolation, not assurance.

**RQ4:** Route predictions do not drift; file-hash and independent certificate replay tests pass. These integrity properties provide observable audit evidence only relative to a trusted reference. They do not prove safety from data/model compromise or distributional shift.

## 6. Discussion

### Why this is a useful negative finding

Stable executable routes prevent post-publication parameter updates from changing earlier task functions, but *immutability does not imply reuse*. The strict gate publishes five distinct encoders for five tasks, so the system resembles a small collection of independent frozen models. The independent baseline matches FAMP on Split and almost matches on Corrupted, and the ungated control matches FAMP exactly. Therefore it would be misleading to attribute accuracy advantages over fine-tuning or replay to the certificate or to reusable abstraction discovery.

### Assurance and utility trade-off

Class-stratified CP bounds combined with Bonferroni spending are transparent and auditable, but strict sample requirements lead to low acceptance for candidates near the tolerated harm boundary. The certificate's excess-risk relation is teacher-relative and assumes an independent acceptance draw for a fixed proposal. Weak teachers, re-used acceptance data, adaptive threshold tuning or distribution shift are outside that assurance contract. A high rejection rate may protect publication decisions while leaving the fallback learner inaccurate and vulnerable to forgetting.

### Engineering cost

Training task-specific units for every task uses additional time and model memory; candidate discovery costs are incurred even if a proposal is rejected. The reported memory proxy is not peak resident memory, actual compressed disk footprint, nor a controlled equal-byte comparison. It is consequently insufficient to claim a cost improvement. A defensible deployment should report these quantities separately, alongside published manifests and audit overhead.

### CPU reduction and external validity

All three benchmark families derive from one small 8×8 image dataset. Rotated and corrupted parity tasks involve repeated source-image indices across domain tasks within each split role. The study cannot be extrapolated to CIFAR-100, class-incremental learning, Transformers, diffusion models, large pretrained backbones, or long task streams. A prior legacy large-backbone FAMP experiment collapsed to chance-level learning and must not be included as positive evidence.

## 7. Threats to validity and reproducibility

1. **Dataset scope:** one small dataset, engineered related domains, known task identity.
2. **Baseline fidelity:** modest in-house EWC/DER++ style adaptations rather than paper-author protocols.
3. **Statistical unit:** two orders within one seed are dependent; n=6 descriptive samples, 3 distinct splits.
4. **Assurance assumptions:** independent acceptance observations, declared class weights, fixed proposal, no repeated acceptance-set peeking, static distribution.
5. **Low bound-violation counts:** zero events with finite trials cannot establish an exact zero population rate.
6. **Integrity model:** self-hashed artifacts require independent trust root for authenticity.
7. **Cost model:** no equal-state budget and no measured CPU peak RSS; memory claim is only a payload proxy.
8. **Method improvement:** the new risk-aware selection heuristic is evaluated as an additional variant, not a mathematically guaranteed selection optimization.
9. **No proven reuse:** no physical sharing occurred for strict FAMP in the executed streams.

## 8. Conclusion

This CPU-scale study implements and tests a transparent, independently replayable rule for publishing immutable executable task routes in continual learning. Its statistical component gives conservative teacher-relative harmful-disagreement certificates under explicit assumptions; a 6,000-trial controlled simulation observed full bound coverage and no unsafe certified acceptance among the 1,800 simulated cases above the harm tolerance. In three scikit-learn Digits-derived task streams and 162 matched training runs, functional routes achieve competitive retention, particularly under digit-pair and rotation sequences. However, strict FAMP matches ungated publication, shows no physical module sharing, uses more retained-state bytes than replay, and is outperformed by DER++-style logit replay on corrupted domains. The evidence therefore supports an **auditable publication protocol and measured safety–utility trade-off**, not a demonstrated new abstraction-learning method or a superior accuracy–memory frontier. The next scientific challenge, outside the claims of this completed artifact, is to produce genuinely transferable units whose certified reuse improves independent-unit and replay controls.

## Background references (verify exact venue metadata before manuscript submission)

- Kirkpatrick et al. (2017), *Overcoming catastrophic forgetting in neural networks*, PNAS — online EWC motivation.
- Buzzega et al. (2020), *Dark Experience for General Continual Learning: a Strong, Simple Baseline*, NeurIPS — replay-with-logits motivation.
- Rebuffi et al. (2017), *iCaRL: Incremental Classifier and Representation Learning*, CVPR — replay-based continual learning background.
- Rusu et al. (2016), *Progressive Neural Networks*, arXiv — immutable modular pathways.
- Clopper and Pearson (1934), *The Use of Confidence or Fiducial Limits Illustrated in the Case of the Binomial*, Biometrika — exact binomial confidence bound.
- Mendez and Eaton (2021), *Lifelong Learning of Compositional Structures*, ICLR — modular reuse.
- Scikit-learn documentation, `load_digits` — bundled small Digits dataset.

**Review/ethics note:** The code and draft used AI assistance. Final conference paper should disclose this according to the CFP, check reference details, and undergo human author approval. Do not represent this markdown draft as an accepted or refereed paper.
