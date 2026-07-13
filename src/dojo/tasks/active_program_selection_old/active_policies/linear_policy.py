from __future__ import annotations

from typing import Any


def linear_policy_num_queries(task: Any, policy: dict[str, Any]) -> int:
    valid_programs = sum(1 for record in task.program_records if record.valid)
    labels_used = len(task.queried_labels)
    label_budget = int(task.cfg.label_budget)

    default_start = policy.get("min_programs_before_query", policy.get("query_every_n_programs", 0))
    start_after_programs = int(policy.get("start_after_programs") or default_start)
    query_every_n_programs = int(policy.get("query_every_n_programs", 0))
    queries_per_interval = int(policy.get("queries_per_interval", policy.get("max_queries_per_step", 1)))
    max_queries_per_step = int(policy.get("max_queries_per_step", queries_per_interval))

    if query_every_n_programs <= 0 or queries_per_interval <= 0 or valid_programs < start_after_programs:
        return 0

    intervals = 1 + (valid_programs - start_after_programs) // query_every_n_programs
    target_labels = min(intervals * queries_per_interval, label_budget)
    return max(0, min(max_queries_per_step, target_labels - labels_used))
