# ViSNN experiment results

_Generated 2026-08-25T01:03:15 from 6 run(s)._

Values are mean +/- std over seeds; single-seed cells show the bare mean. Empty cells were not reported by those runs.

## Track A - depth estimation

| group | runs | val_rmse | val_mae | val_abs_rel | val_sq_rel | val_rmse_log | val_delta1 | val_delta2 | val_delta3 | best_metric | spike_rate | sops_per_frame | snn_joules_per_frame |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `depth_fnmtn_L8_lam0.25_ep2_n32_cb2_syn1` | 1 | 28.7270 | 27.1961 | 0.8153 | 22.7080 | 1.7626 | 0.0053 | 0.0053 | 0.0053 | 28.7270 | 0.5187 | 674652920.5625 | 0.0007 |
| `depth_cont_ep1_n32_syn1` | 1 | 29.1742 | 27.6031 | 0.8269 | 23.3956 | 1.8404 | 0.0053 | 0.0053 | 0.0053 | 29.1742 | 0.0000 | 0.0000 | 0.0000 |
| `depth` | 1 |  |  |  |  |  |  |  |  | 29.2448 | 0.0533 |  |  |
| `depth_fnmtn_L8_lam0.1_ep1_n16_cb2_syn1` | 1 | 29.2569 | 27.6836 | 0.8293 | 23.5320 | 1.8568 | 0.0053 | 0.0053 | 0.0053 | 29.2569 | 0.6022 | 760812058.7500 | 0.0008 |

Seeds present: 42, 42, 42, 42.

## Track B - SSD detection

| group | runs | mAP | mAP@[.5:.95] | num_classes_evaluated | best_metric | spike_rate | sops_per_frame | snn_joules_per_frame |
|---|---|---|---|---|---|---|---|---|
| `ssd_fnmtn_L8_lam0.25_ep2_n16_cb2_syn1` | 1 | 0.0129 |  | 3.0000 | 0.0144 | 0.4787 | 367926277.1250 | 0.0004 |
| `ssd` | 1 |  |  |  | 0.0000 | 0.2047 |  |  |

Seeds present: 42, 42.

