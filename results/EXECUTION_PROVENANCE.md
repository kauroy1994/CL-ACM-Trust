# Completed CPU experiment provenance

- Execution date: 2026-10-30.
- Protocol: trustcl-cpu-1.0.
- Python: 3.13.5, Linux x86_64 CPU.
- Source: Python modules in this repository.
- Fixed training: epochs=30, seeds=(11,23,37), task orders=(0,1), methods=(finetune,replay,derpp,ewc,independent,ungated,famp,no_response,famp_naive).
- Datasets: scikit-learn Digits-derived split/rotated/corrupted tasks, five task IDs each.
- Full neural campaign: 3×3×2×9 = 162 completed five-task streams; all experiment records were saved as per-run JSON before aggregation. For a fresh copy of the records, run the documented CPU command. Every one of the 27 dataset/method groups contains six runs.
- Calibration: 10 known-risk levels × 120 trials × five task indices = 6,000 independent simulated acceptance decisions; n_per_class=100, epsilon=.25, total alpha=.05.
- Local tests: pytest core suite 5/5 passed.
- Aggregate saved evidence: benchmark_summary.csv (27 rows), paired_effects.csv (18 rows), admission_summary.csv (15 rows), calibration_summary.csv (10 rows). Floating point aggregates may be rounded for readable version-controlled CSV; regenerate unrounded values with the report script.
- The two task orders within a data seed are correlated; six streams do not imply six independent data splits. Confidence intervals are descriptive.
- This is not a GPU execution, an ImageNet/CIFAR/MNIST benchmark, an equal-byte comparison, or a measured production throughput test. CPU wall times include platform noise. The cost metric is model/replay payload plus metadata estimate, not peak RAM.
- No physical functional-unit sharing was observed for strict FAMP. No statistical-learning improvement from gating over ungated modular publication was demonstrated.
- Original legacy FAMP GPU/Colab results are separate and must never be merged into these aggregates.
