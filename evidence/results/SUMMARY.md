# Reweighted posterior vs full refit, by Pareto k-hat


## priors: 101 scenarios, 98 with a converged refit

Refit noise (median error between two refits of one scenario, refit SDs): 0.009 (8 scenarios); split-chain noise 0.013. Refits with divergences: 13 (total 31 of 1212000 draws).


By k-hat (error in refit posterior SDs; `do_nothing` = unweighted reference posterior):

| k_bin | n | ess_median | err_median | err_max_median | err_max_p90 | do_nothing_max_median | closer_worst |
| --- | --- | --- | --- | --- | --- | --- | --- |
| <0.3 | 20 | 0.95 | 0.01 | 0.027 | 0.039 | 0.227 | 0.9 |
| 0.3-0.5 | 20 | 0.406 | 0.014 | 0.043 | 0.11 | 0.669 | 1 |
| 0.5-0.7 | 20 | 0.084 | 0.029 | 0.084 | 0.357 | 1.156 | 1 |
| 0.7-1 | 19 | 0.014 | 0.072 | 0.238 | 0.574 | 1.656 | 1 |
| >=1 | 19 | 0.002 | 0.15 | 0.942 | 2.127 | 5.052 | 1 |


By k-hat, per posterior quantity (median error in refit SDs). `moved`: the change shifts the quantity by more than 0.1 SD (do-nothing error); `share_closer`: share of quantities where the reweighted mean is closer to the refit than the unweighted reference:

| k_bin | quantities | n | err_reweighted | err_do_nothing | share_closer |
| --- | --- | --- | --- | --- | --- |
| <0.3 | not moved | 135 | 0.009 | 0.017 | 0.696 |
| <0.3 | moved | 45 | 0.015 | 0.275 | 1 |
| 0.3-0.5 | not moved | 90 | 0.011 | 0.031 | 0.744 |
| 0.3-0.5 | moved | 90 | 0.019 | 0.343 | 1 |
| 0.5-0.7 | not moved | 78 | 0.03 | 0.042 | 0.615 |
| 0.5-0.7 | moved | 102 | 0.034 | 0.547 | 0.99 |
| 0.7-1 | not moved | 47 | 0.075 | 0.05 | 0.383 |
| 0.7-1 | moved | 124 | 0.076 | 0.724 | 0.96 |
| >=1 | not moved | 40 | 0.078 | 0.049 | 0.35 |
| >=1 | moved | 131 | 0.175 | 0.859 | 0.931 |


Scenarios with k-hat < 0.7, by ESS fraction:

| ess_bin | n | ess_median | err_median | err_max_median | err_max_p90 | do_nothing_max_median | closer_worst |
| --- | --- | --- | --- | --- | --- | --- | --- |
| <5% | 6 | 0.035 | 0.066 | 0.231 | 0.39 | 1.732 | 1 |
| 5-20% | 13 | 0.085 | 0.029 | 0.082 | 0.281 | 1.158 | 1 |
| >20% | 41 | 0.577 | 0.012 | 0.031 | 0.06 | 0.596 | 0.951 |


Refits that did not converge (excluded):

| id | pareto_k | ess_fraction | refit_max_r_hat | refit_min_ess_bulk |
| --- | --- | --- | --- | --- |
| p_business_none | 0.966 | 0.006 | 1.614 | 9.881 |
| p_business_064 | 0.79 | 0.006 | 1.376 | 13.041 |
| p_joint_084 | 1.114 | 0.002 | 1.376 | 12.982 |


## likelihood: 43 scenarios, 43 with a converged refit

Refit noise (median error between two refits of one scenario, refit SDs): 0.007 (7 scenarios); split-chain noise 0.013. Refits with divergences: 1 (total 1 of 516000 draws).


By k-hat (error in refit posterior SDs; `do_nothing` = unweighted reference posterior):

| k_bin | n | ess_median | err_median | err_max_median | err_max_p90 | do_nothing_max_median | closer_worst |
| --- | --- | --- | --- | --- | --- | --- | --- |
| <0.3 | 8 | 0.956 | 0.007 | 0.017 | 0.033 | 0.109 | 0.875 |
| 0.3-0.5 | 8 | 0.554 | 0.011 | 0.026 | 0.049 | 0.406 | 1 |
| 0.5-0.7 | 8 | 0.115 | 0.019 | 0.057 | 0.247 | 0.699 | 1 |
| 0.7-1 | 9 | 0.01 | 0.121 | 0.323 | 1.162 | 1.383 | 1 |
| >=1 | 10 | 0 | 1.051 | 3.354 | 4.268 | 5.069 | 1 |


By k-hat, per posterior quantity (median error in refit SDs). `moved`: the change shifts the quantity by more than 0.1 SD (do-nothing error); `share_closer`: share of quantities where the reweighted mean is closer to the refit than the unweighted reference:

| k_bin | quantities | n | err_reweighted | err_do_nothing | share_closer |
| --- | --- | --- | --- | --- | --- |
| <0.3 | not moved | 58 | 0.008 | 0.01 | 0.569 |
| <0.3 | moved | 14 | 0.004 | 0.172 | 1 |
| 0.3-0.5 | not moved | 40 | 0.009 | 0.021 | 0.65 |
| 0.3-0.5 | moved | 32 | 0.013 | 0.283 | 1 |
| 0.5-0.7 | not moved | 47 | 0.016 | 0.02 | 0.574 |
| 0.5-0.7 | moved | 25 | 0.04 | 0.388 | 1 |
| 0.7-1 | not moved | 43 | 0.053 | 0.04 | 0.326 |
| 0.7-1 | moved | 38 | 0.246 | 0.539 | 0.816 |
| >=1 | not moved | 16 | 0.31 | 0.058 | 0.125 |
| >=1 | moved | 74 | 1.031 | 0.887 | 0.568 |


Scenarios with k-hat < 0.7, by ESS fraction:

| ess_bin | n | ess_median | err_median | err_max_median | err_max_p90 | do_nothing_max_median | closer_worst |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5-20% | 7 | 0.091 | 0.023 | 0.066 | 0.252 | 0.735 | 1 |
| >20% | 17 | 0.742 | 0.008 | 0.024 | 0.036 | 0.269 | 0.941 |
