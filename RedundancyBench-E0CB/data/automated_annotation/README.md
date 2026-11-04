# Annotated model-generated trajectories

This directory contains 1,731 labeled trajectories from five trajectory-generation models. Three separate models (`deepseek-v4-flash`, `glm-5.3-flash`, and `gpt-5.6-luna`) annotated redundancy; a trajectory's annotation was retained when at least two agreed on the redundant-step indices, types, and evidence indices. These are **model-consensus reference labels, not individually human-verified Gold**. The source-provided `label_source` and `consensus_annotators` fields retain that provenance.

| Domain | Trajectories | Annotations | Distinct tasks | Labeled steps |
| --- | ---: | ---: | ---: | ---: |
| airline | 165 | 165 | 41 | 483 |
| retail | 412 | 412 | 98 | 744 |
| telecom | 1,154 | 1,154 | 114 | 1,729 |
| Total | 1,731 | 1,731 | — | 2,956 |

As in `human_annotation/`, each domain has `final_traces.json` with `tasks` and `simulations` arrays and a separate `annotation.json` array. Every labeled trajectory has one simulation and one annotation. Task definitions are included once per distinct `task_id` in each domain; multiple simulations may share a task.

Join simulations and annotations by `trajectory_id`, **not** by `task_id` or array position. Each ID has the form `source_id:domain:original_sim_index`. `source_id` identifies the trajectory-generating model; `sim_index` preserves its original position in that model's source file. All original annotation fields, including `reason`, `confidence`, consensus fields, and `label_source`, are retained for provenance. The upstream source had already applied the two-of-three consensus and positive-trajectory selection; this release added no further confidence or type filter. All 1,731 trajectories have at least one labeled redundant step.

Message order and all prompt-visible message fields, together with the reference step/evidence/type arrays, match the original submissions. Nonessential run metadata such as per-message usage and cost is omitted. The 94 Telecom simulations missing a policy in the raw files use the policy restored and audited during preparation; all other policies are unchanged.
