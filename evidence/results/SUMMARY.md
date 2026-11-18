# Reweighted posterior vs full refit, by Pareto k-hat


## priors: 101 scenarios, 101 with a converged refit

Refit noise (median error between two refits of one scenario, refit SDs): 0.009 (9 scenarios); split-chain noise 0.013. Refits with divergences: 13 (total 31 of 1212000 draws).


By k-hat (error in refit posterior SDs; `do_nothing` = unweighted reference posterior):

| k_bin | n | ess_median | err_median | err_max_median | err_max_p90 | err_q_max_median | sd_off_max_median | do_nothing_max_median | closer_worst |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| <0.3 | 20 | 0.95 | 0.01 | 0.027 | 0.039 | 0.077 | 0.027 | 0.227 | 0.9 |
| 0.3-0.5 | 20 | 0.406 | 0.014 | 0.043 | 0.11 | 0.14 | 0.038 | 0.669 | 1 |
| 0.5-0.7 | 20 | 0.084 | 0.029 | 0.084 | 0.357 | 0.301 | 0.1 | 1.156 | 1 |
| 0.7-1 | 21 | 0.01 | 0.083 | 0.346 | 0.573 | 1.05 | 0.226 | 1.783 | 1 |
| >=1 | 20 | 0.002 | 0.156 | 0.928 | 1.984 | 2.125 | 0.466 | 4.998 | 1 |


By k-hat, per posterior quantity (median error in refit SDs). `moved`: the change shifts the quantity by more than 0.1 SD (do-nothing error); `share_closer`: share of quantities where the reweighted mean is closer to the refit than the unweighted reference:

| k_bin | quantities | n | err_reweighted | err_do_nothing | share_closer |
| --- | --- | --- | --- | --- | --- |
| <0.3 | not moved | 135 | 0.009 | 0.017 | 0.696 |
| <0.3 | moved | 45 | 0.015 | 0.275 | 1 |
| 0.3-0.5 | not moved | 90 | 0.011 | 0.031 | 0.744 |
| 0.3-0.5 | moved | 90 | 0.019 | 0.343 | 1 |
| 0.5-0.7 | not moved | 78 | 0.03 | 0.042 | 0.615 |
| 0.5-0.7 | moved | 102 | 0.034 | 0.547 | 0.99 |
| 0.7-1 | not moved | 50 | 0.082 | 0.049 | 0.36 |
| 0.7-1 | moved | 139 | 0.077 | 0.691 | 0.935 |
| >=1 | not moved | 44 | 0.078 | 0.047 | 0.341 |
| >=1 | moved | 136 | 0.178 | 0.842 | 0.919 |


By verdict of `psis_reweight` (k-hat, ESS >= 400, ESS >= 5% for "use it"):

| verdict | n | ess_median | err_median | err_max_median | err_max_p90 | err_q_max_median | sd_off_max_median | do_nothing_max_median | closer_worst |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| use it | 38 | 0.648 | 0.012 | 0.031 | 0.064 | 0.091 | 0.03 | 0.511 | 0.947 |
| check | 19 | 0.085 | 0.029 | 0.082 | 0.304 | 0.278 | 0.097 | 1.158 | 1 |
| refit | 44 | 0.004 | 0.099 | 0.518 | 1.416 | 1.374 | 0.34 | 3.146 | 1 |


Scenarios with k-hat < 0.7, by ESS fraction:

| ess_bin | n | ess_median | err_median | err_max_median | err_max_p90 | err_q_max_median | sd_off_max_median | do_nothing_max_median | closer_worst |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| <5% | 6 | 0.035 | 0.066 | 0.231 | 0.39 | 0.647 | 0.179 | 1.732 | 1 |
| 5-20% | 13 | 0.085 | 0.029 | 0.082 | 0.281 | 0.278 | 0.098 | 1.158 | 1 |
| >20% | 41 | 0.577 | 0.012 | 0.031 | 0.06 | 0.094 | 0.032 | 0.596 | 0.951 |


Refit A did not converge; replaced by a converged refit with another seed:

| id | refit_source | refit_seed | refit_max_r_hat | refit_min_ess_bulk |
| --- | --- | --- | --- | --- |
| p_business_none | rerun_priors_E | 50001 | 1.002 | 3080.28 |
| p_business_064 | rerun_priors_C | 30065 | 1.002 | 3350.54 |
| p_joint_084 | rerun_priors_C | 30085 | 1.003 | 2916.34 |


## likelihood: 46 scenarios, 46 with a converged refit

Refit noise (median error between two refits of one scenario, refit SDs): 0.007 (10 scenarios); split-chain noise 0.013. Refits with divergences: 1 (total 1 of 552000 draws).


By k-hat (error in refit posterior SDs; `do_nothing` = unweighted reference posterior):

| k_bin | n | ess_median | err_median | err_max_median | err_max_p90 | err_q_max_median | sd_off_max_median | do_nothing_max_median | closer_worst |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| <0.3 | 9 | 0.942 | 0.007 | 0.017 | 0.043 | 0.108 | 0.027 | 0.12 | 0.889 |
| 0.3-0.5 | 9 | 0.521 | 0.011 | 0.026 | 0.086 | 0.137 | 0.043 | 0.406 | 1 |
| 0.5-0.7 | 8 | 0.115 | 0.019 | 0.057 | 0.247 | 0.168 | 0.036 | 0.699 | 1 |
| 0.7-1 | 9 | 0.01 | 0.121 | 0.323 | 1.162 | 1.024 | 0.207 | 1.383 | 1 |
| >=1 | 11 | 0 | 1.05 | 3.34 | 4.184 | 4.212 | 0.511 | 5.174 | 1 |


By k-hat, per posterior quantity (median error in refit SDs). `moved`: the change shifts the quantity by more than 0.1 SD (do-nothing error); `share_closer`: share of quantities where the reweighted mean is closer to the refit than the unweighted reference:

| k_bin | quantities | n | err_reweighted | err_do_nothing | share_closer |
| --- | --- | --- | --- | --- | --- |
| <0.3 | not moved | 63 | 0.008 | 0.01 | 0.54 |
| <0.3 | moved | 18 | 0.005 | 0.231 | 1 |
| 0.3-0.5 | not moved | 45 | 0.01 | 0.021 | 0.578 |
| 0.3-0.5 | moved | 36 | 0.014 | 0.325 | 1 |
| 0.5-0.7 | not moved | 47 | 0.016 | 0.02 | 0.574 |
| 0.5-0.7 | moved | 25 | 0.04 | 0.388 | 1 |
| 0.7-1 | not moved | 43 | 0.053 | 0.04 | 0.326 |
| 0.7-1 | moved | 38 | 0.246 | 0.539 | 0.816 |
| >=1 | not moved | 21 | 0.211 | 0.039 | 0.095 |
| >=1 | moved | 78 | 0.977 | 1.002 | 0.59 |


By verdict of `psis_reweight` (k-hat, ESS >= 400, ESS >= 5% for "use it"):

| verdict | n | ess_median | err_median | err_max_median | err_max_p90 | err_q_max_median | sd_off_max_median | do_nothing_max_median | closer_worst |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| use it | 17 | 0.742 | 0.008 | 0.024 | 0.053 | 0.108 | 0.027 | 0.269 | 0.941 |
| check | 8 | 0.115 | 0.019 | 0.057 | 0.247 | 0.168 | 0.036 | 0.699 | 1 |
| refit | 21 | 0.002 | 0.162 | 0.546 | 4.085 | 1.37 | 0.313 | 4.413 | 1 |


Scenarios with k-hat < 0.7, by ESS fraction:

| ess_bin | n | ess_median | err_median | err_max_median | err_max_p90 | err_q_max_median | sd_off_max_median | do_nothing_max_median | closer_worst |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| <5% | 1 | 0.014 | 0.074 | 0.132 | 0.132 | 0.564 | 0.13 | 5.529 | 1 |
| 5-20% | 8 | 0.081 | 0.028 | 0.071 | 0.247 | 0.235 | 0.066 | 0.77 | 1 |
| >20% | 17 | 0.742 | 0.008 | 0.024 | 0.036 | 0.104 | 0.026 | 0.269 | 0.941 |
