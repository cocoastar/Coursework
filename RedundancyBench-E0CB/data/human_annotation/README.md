# Human-annotated trajectories

This directory contains the positive (redundancy-present) subset of the original 193 human-annotated trajectories.

| Domain | Trajectories | Annotations |
| --- | ---: | ---: |
| airline | 38 | 38 |
| retail | 40 | 40 |
| telecom | 111 | 111 |
| Total | 189 | 189 |

Each domain has `tasks` and `simulations` arrays in `final_traces.json`, plus Gold annotations in `annotation.json`. These three collections correspond one-to-one by `task_id` within each domain.

The four Retail records with empty `redundant_step_idx` arrays (task IDs `10`, `61`, `96`, and `101`) were excluded from all three collections. The release preserves message order, all prompt-visible message fields, policies, and Gold annotations. It omits provider session IDs, timestamps, costs, raw API/audio metadata, and task issue metadata. The source files were not modified.
