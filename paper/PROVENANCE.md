# Evidence and provenance notes

**Executed analysis:** CPU-only model training and synthetic assurance calibration were completed locally prior to the GitHub deposit on 2026-10-30. This repository was created from the preserved `/mnt/data/cl_trust_repo` snapshot; the logged neural campaign has 162 task-stream JSON records (three seeds, two orders, three datasets, nine methods), each containing five stage logs. The archived data include the full per-run JSON files, admission traces, audit CSV, summary CSV, and risk-coverage and accuracy plots. No new neural training was performed during GitHub publication.

**Source revision:** `trustcl-cpu-1.0`. Dependencies in `requirements.txt`. Baseline adaptations trained with 30 epochs per task stage. Software versions are not locked down to individual wheel digests and environment-dependent timing is descriptive.

**Calibrations:** `results/calibration/calibration_trials.csv` has 6,000 task-level records, constructed from 120 repetitions × 5 tasks × 10 known-risk levels; 100 acceptance examples per class. The theoretical finite-sample coverage assertion is conditional and applies to each preselected proposal given the sampling assumptions. Monte Carlo coverage is descriptive rather than a proof.

**Data benchmark:** scikit-learn `load_digits` (8×8). Split Digits is five paired-digit tasks; Rotated/Corrupted Digits are parity-classification tasks with transformations on original images. Train, selection, acceptance and test roles were globally partitioned by image ID before domain transformations. The same underlying original images recur across domains *within the same role*, so transformed tasks are not statistically independent.

**Risk and integrity warnings:** no authenticated manifest signature, no independent empirical certification of the teacher, no input-intervention robustness guarantee, no equal-persistent-memory evaluation. Raw acceptance predictions for each neural run are not included in JSON, so full external re-verification needs to reconstruct them from the fixed source/dataset and candidate models; the auditor implementation can reverify a certificate once prediction evidence is supplied.

**Prior exploratory evidence:** legacy MLSys FAMP computations and a failed ResNet18GN CIFAR-100 track were separate versions and are not pooled into TRUST CPU benchmark results. The core paper must not claim these are validated by this repository.

**Double-blind issue:** the repository is directly associated with an author's account. Do not cite this public author-linked repository in an anonymized manuscript unless the venue permits it; prepare an anonymized artifact route as needed.
