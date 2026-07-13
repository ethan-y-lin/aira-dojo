from __future__ import annotations

from typing import Any

from dojo.tasks.active_program_selection.active_policies.linear_policy import (
    linear_policy_num_queries,
)


def active_policy_num_queries(task: Any) -> int:
    policy = task.active_policy()
    name = policy.get("name", "disabled")
    if name in {"disabled", "none", None}:
        return 0
    if name in {"linear", "query_every_n_programs"}:
        return linear_policy_num_queries(task, policy)
    raise NotImplementedError(f"Unsupported active policy: {name}")
