# Executed CPU benchmark results

All values below are automatically derived from completed `*.json` task streams. Each dataset/method cell has 3 seeds × 2 orders (n=6). No GPU experiments are claimed.

## Final accuracy, forgetting, published units and resource costs

| dataset          | method      |   n |   accuracy_mean |   accuracy_sd |   forgetting_mean |   acquisition_mean |   admitted_mean |   shared_units_mean |   route_drift_total |   memory_kib_mean |   train_seconds_mean |
|:-----------------|:------------|----:|----------------:|--------------:|------------------:|-------------------:|----------------:|--------------------:|--------------------:|------------------:|---------------------:|
| corrupted_digits | derpp       |   6 |          0.9732 |        0.016  |            0.0012 |             0.9707 |          0      |              0      |                   0 |           55.7871 |               1.6723 |
| corrupted_digits | ewc         |   6 |          0.97   |        0.0145 |            0.0035 |             0.9687 |          0      |              0      |                   0 |           68.3633 |               1.5919 |
| corrupted_digits | famp        |   6 |          0.96   |        0.023  |            0      |             0.96   |          5      |              0      |                   0 |           74.918  |               2.208  |
| corrupted_digits | famp_naive  |   6 |          0.9671 |        0.0177 |            0.0027 |             0.9682 |          1.8333 |              0      |                   0 |           62.2108 |               2.3884 |
| corrupted_digits | finetune    |   6 |          0.969  |        0.0126 |            0.0046 |             0.9694 |          0      |              0      |                   0 |           22.9775 |               1.2475 |
| corrupted_digits | independent |   6 |          0.9602 |        0.0243 |            0      |             0.9602 |          5      |              0      |                   0 |           74.9248 |               2.3468 |
| corrupted_digits | no_response |   6 |          0.9587 |        0.0232 |            0      |             0.9587 |          5      |              1      |                   0 |           33.9092 |               2.0611 |
| corrupted_digits | replay      |   6 |          0.9713 |        0.0175 |            0.0046 |             0.9713 |          0      |              0      |                   0 |           54.8506 |               1.5192 |
| corrupted_digits | ungated     |   6 |          0.96   |        0.023  |            0      |             0.96   |          5      |              0      |                   0 |           74.9209 |               2.0759 |
| rotated_digits   | derpp       |   6 |          0.9336 |        0.0123 |            0.0305 |             0.963  |          0      |              0      |                   0 |           55.7852 |               1.8678 |
| rotated_digits   | ewc         |   6 |          0.8626 |        0.0451 |            0.0966 |             0.9589 |          0      |              0      |                   0 |           68.3613 |               1.6464 |
| rotated_digits   | famp        |   6 |          0.9547 |        0.022  |            0      |             0.9547 |          5      |              0      |                   0 |           74.916  |               2.3022 |
| rotated_digits   | famp_naive  |   6 |          0.9547 |        0.022  |            0      |             0.9547 |          5      |              0      |                   0 |           74.9219 |               2.2585 |
| rotated_digits   | finetune    |   6 |          0.7806 |        0.0447 |            0.1845 |             0.9651 |          0      |              0      |                   0 |           22.9756 |               1.2738 |
| rotated_digits   | independent |   6 |          0.9484 |        0.0213 |            0.0043 |             0.9526 |          4.6667 |              0      |                   0 |           73.5846 |               2.2676 |
| rotated_digits   | no_response |   6 |          0.9525 |        0.0206 |            0.0016 |             0.9541 |          4.8333 |              0      |                   0 |           74.2537 |               2.2526 |
| rotated_digits   | replay      |   6 |          0.8956 |        0.0193 |            0.069  |             0.9646 |          0      |              0      |                   0 |           54.8486 |               1.5109 |
| rotated_digits   | ungated     |   6 |          0.9547 |        0.022  |            0      |             0.9547 |          5      |              0      |                   0 |           74.9189 |               2.1923 |
| split_digits     | derpp       |   6 |          0.9644 |        0.0182 |            0.0178 |             0.9771 |          0      |              0      |                   0 |           55.7832 |               0.4767 |
| split_digits     | ewc         |   6 |          0.799  |        0.0274 |            0.1899 |             0.9883 |          0      |              0      |                   0 |           68.3594 |               0.4089 |
| split_digits     | famp        |   6 |          0.9922 |        0.0062 |            0      |             0.9922 |          5      |              0      |                   0 |           74.9141 |               0.5452 |
| split_digits     | famp_naive  |   6 |          0.9819 |        0.0147 |            0.008  |             0.9893 |          2.5    |              0      |                   0 |           64.8833 |               0.5904 |
| split_digits     | finetune    |   6 |          0.7919 |        0.0208 |            0.1975 |             0.9894 |          0      |              0      |                   0 |           22.9736 |               0.3964 |
| split_digits     | independent |   6 |          0.9922 |        0.0062 |            0      |             0.9922 |          5      |              0      |                   0 |           74.9209 |               0.5206 |
| split_digits     | no_response |   6 |          0.9827 |        0.0132 |            0.0019 |             0.9845 |          4.3333 |              1.1667 |                   0 |           51.7367 |               0.5612 |
| split_digits     | replay      |   6 |          0.9529 |        0.0123 |            0.0372 |             0.9895 |          0      |              0      |                   0 |           54.8467 |               0.3866 |
| split_digits     | ungated     |   6 |          0.9922 |        0.0062 |            0      |             0.9922 |          5      |              0      |                   0 |           74.917  |               0.5421 |

## Paired FAMP differences (percentage points)

| dataset          | versus      |   mean_accuracy_difference_pp |   bootstrap_lo_pp |   bootstrap_hi_pp |   pairs |
|:-----------------|:------------|------------------------------:|------------------:|------------------:|--------:|
| corrupted_digits | finetune    |                        -0.898 |            -1.841 |            -0.056 |       6 |
| corrupted_digits | replay      |                        -1.122 |            -1.695 |            -0.561 |       6 |
| corrupted_digits | independent |                        -0.011 |            -0.213 |             0.224 |       6 |
| corrupted_digits | ungated     |                         0     |             0     |             0     |       6 |
| corrupted_digits | famp_naive  |                        -0.707 |            -1.168 |            -0.247 |       6 |
| corrupted_digits | no_response |                         0.135 |            -0.079 |             0.348 |       6 |
| rotated_digits   | finetune    |                        17.407 |            12.806 |            20.943 |       6 |
| rotated_digits   | replay      |                         5.903 |             4.747 |             7.194 |       6 |
| rotated_digits   | independent |                         0.629 |             0.224 |             1.19  |       6 |
| rotated_digits   | ungated     |                         0     |             0     |             0     |       6 |
| rotated_digits   | famp_naive  |                         0     |             0     |             0     |       6 |
| rotated_digits   | no_response |                         0.213 |             0     |             0.64  |       6 |
| split_digits     | finetune    |                        20.031 |            18.76  |            21.32  |       6 |
| split_digits     | replay      |                         3.934 |             2.815 |             4.86  |       6 |
| split_digits     | independent |                         0     |            -0.167 |             0.167 |       6 |
| split_digits     | ungated     |                         0     |             0     |             0     |       6 |
| split_digits     | famp_naive  |                         1.031 |             0.337 |             1.922 |       6 |
| split_digits     | no_response |                         0.956 |             0.263 |             1.626 |       6 |

## Admission inspection

| dataset          | method      |   selected_proposals |   acceptance_rate |   mean_harm_upper |   mean_test_harm |   observed_test_exceedances |
|:-----------------|:------------|---------------------:|------------------:|------------------:|-----------------:|----------------------------:|
| corrupted_digits | famp        |                   30 |            1      |            0.0546 |           0.0182 |                           0 |
| corrupted_digits | famp_naive  |                   30 |            0.3667 |            0.0476 |           0.0172 |                           0 |
| corrupted_digits | independent |                   30 |            1      |            0.0527 |           0.018  |                           0 |
| corrupted_digits | no_response |                   30 |            1      |            0.047  |           0.0178 |                           0 |
| corrupted_digits | ungated     |                   30 |            1      |            0.0546 |           0.0182 |                           0 |
| rotated_digits   | famp        |                   30 |            1      |            0.0558 |           0.0253 |                           0 |
| rotated_digits   | famp_naive  |                   30 |            1      |            0.0558 |           0.0253 |                           0 |
| rotated_digits   | independent |                   30 |            0.9333 |            0.058  |           0.0287 |                           0 |
| rotated_digits   | no_response |                   30 |            0.9667 |            0.06   |           0.0282 |                           0 |
| rotated_digits   | ungated     |                   30 |            1      |            0.0558 |           0.0253 |                           0 |
| split_digits     | famp        |                   30 |            1      |            0.0757 |           0.0043 |                           0 |
| split_digits     | famp_naive  |                   30 |            0.5    |            0.0916 |           0.0133 |                           0 |
| split_digits     | independent |                   30 |            1      |            0.0745 |           0.0043 |                           0 |
| split_digits     | no_response |                   30 |            0.8667 |            0.0948 |           0.0161 |                           0 |
| split_digits     | ungated     |                   30 |            1      |            0.0757 |           0.0043 |                           0 |

**Interpretation:** In this small CPU protocol, strict FAMP is not demonstrably more accurate or more memory efficient than independent task units. Reuse may be absent when the response gate is stringent. Apparent test harm uses small held-out sets and is not the population truth. Bootstrap intervals are descriptive, with just three independent seeds, and should not be treated as formal hypothesis tests.
