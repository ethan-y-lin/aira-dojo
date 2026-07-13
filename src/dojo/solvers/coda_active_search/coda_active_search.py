from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import torch

from dojo.config_dataclasses.solver.coda_active_search import (
    CODAActiveSearchSolverConfig,
)
from dojo.core.solvers.utils.journal import Node
from dojo.core.solvers.utils.metric import MetricValue, WorstMetricValue
from dojo.core.solvers.utils.response import extract_code
from dojo.core.solvers.utils.search_exporter import export_search_results
from dojo.core.tasks.constants import EXECUTION_OUTPUT, VALID_SOLUTION
from dojo.solvers.greedy.greedy import Greedy


class CODAActiveSearch(Greedy):
    """Active program search with a CODA uncertainty-diversity controller."""

    def __init__(self, cfg: CODAActiveSearchSolverConfig, task_info):
        super().__init__(cfg, task_info=task_info)
        self.cfg: CODAActiveSearchSolverConfig = cfg
        self.controller_iteration = 0

    def __call__(self, task, state):
        self.logger.info("Starting CODA active program search")
        self.create_root_node()

        while self._can_continue(task):
            start_time = time.monotonic()
            state, _ = self.controller_step(task, state)
            self.state.running_time += time.monotonic() - start_time
            self.controller_iteration += 1

            self.logger.info(
                f"Controller iter {self.controller_iteration}: "
                f"time taken so far {self.state.running_time:.3f} seconds"
            )
            self.save_checkpoint()

            if self.state.running_time >= self.cfg.time_limit_secs:
                self.logger.info("Maximum runtime reached, stopping search")
                break

        best_node = self.journal.get_best_node()
        try:
            export_search_results(self.cfg, self.journal, self.logger, "CODAActiveSearch")
        except Exception as e:
            self.logger.error(f"Error exporting search results: {e}")

        if best_node:
            return state, best_node.code, best_node
        self.logger.info("No suitable code found after active search.")
        return state, None, None

    def controller_step(self, task, state):
        if not self.journal.nodes or self.data_preview is None:
            self.update_data_preview(state)

        action_info = self._select_action(task)
        action = action_info["action"]
        self.logger.info(f"CODA controller selected action: {action_info}")

        if action == "QueryLabel":
            event = task.query_label()
            task.refresh_node_fitness(self.journal)
            self._record_controller_event(task, {**action_info, "query_event": event})
            self._log_state()
            return state, {"active_event": event}

        if action == "Tune":
            parent_node = action_info["target_node"]
            result_node = self._improve(parent_node)
            result_node.operators_used = ["tune" if op == "improve" else op for op in result_node.operators_used]
        else:
            result_node = self._draft()

        self._record_controller_event(task, self._serializable_action_info(action_info))
        state, eval_result = self._evaluate_and_record(task, state, result_node)
        if result_node.is_buggy:
            state, eval_result = self._debug_until_valid(task, state, result_node, eval_result)
        self._log_state()
        return state, eval_result

    def _select_action(self, task) -> dict[str, Any]:
        if not self._has_program_budget(task):
            return {
                "action": "QueryLabel",
                "reason": "program_budget_exhausted",
                "valid_programs": task.valid_program_count(),
                "label_budget_remaining": task.label_budget_remaining(),
            }

        if task.valid_program_count() < self.cfg.initial_drafts:
            return {
                "action": "Draft",
                "reason": "initial_drafts",
                "valid_programs": task.valid_program_count(),
                "initial_drafts": self.cfg.initial_drafts,
            }

        allowed_ids = self._eligible_program_ids()
        controller_state = task.controller_state(allowed_program_ids=allowed_ids)
        uncertainty = float(controller_state["uncertainty"])
        diversity = float(controller_state["diversity"])

        if uncertainty < self.cfg.uncertainty_threshold:
            target_program_id, target_node = self._sample_tune_target(controller_state)
            return {
                "action": "Tune",
                "reason": "low_uncertainty",
                "uncertainty": uncertainty,
                "diversity": diversity,
                "target_program_id": target_program_id,
                "target_node": target_node,
                "tune_temperature": self.cfg.tune_temperature,
            }

        if diversity < self.cfg.diversity_threshold or task.label_budget_remaining() <= 0:
            return {
                "action": "Draft",
                "reason": "high_uncertainty_low_diversity_or_no_labels",
                "uncertainty": uncertainty,
                "diversity": diversity,
            }

        return {
            "action": "QueryLabel",
            "reason": "high_uncertainty_high_diversity",
            "uncertainty": uncertainty,
            "diversity": diversity,
        }

    def _sample_tune_target(self, controller_state: dict[str, Any]) -> tuple[int, Node]:
        program_ids = list(controller_state["program_ids"])
        pbest = torch.tensor(controller_state["pbest"], dtype=torch.float32)
        if not program_ids or pbest.numel() == 0:
            best = self.journal.get_best_node()
            if best is None:
                raise RuntimeError("Cannot tune because there are no valid programs.")
            return int(best.metric.info["active_program_id"]), best

        weights = pbest.clamp_min(1e-12).pow(1.0 / self.cfg.tune_temperature)
        weights = weights / weights.sum().clamp_min(1e-12)
        local_idx = int(torch.multinomial(weights, num_samples=1).item())
        program_id = int(program_ids[local_idx])
        node = self._node_for_program_id(program_id)
        if node is None:
            best = self.journal.get_best_node()
            if best is None:
                raise RuntimeError(f"CODA selected program {program_id}, but no journal node was found.")
            return int(best.metric.info["active_program_id"]), best
        return program_id, node

    def _evaluate_and_record(self, task, state, node: Node):
        self.logger.debug(f"Executing generated code for action {node.operators_used}")
        state, eval_result = task.step_task(state, extract_code(node.code))
        self.parse_eval_result(node=node, eval_result=eval_result)
        self.journal.append(node)
        self.state.current_step = len(self.journal.nodes)
        if hasattr(task, "refresh_node_fitness"):
            task.refresh_node_fitness(self.journal)
        return state, eval_result

    def _debug_until_valid(self, task, state, node: Node, eval_result: dict[str, Any]):
        attempts = 0
        current = node
        while current.is_buggy and attempts < self.cfg.max_debug_attempts and self._has_program_budget(task):
            attempts += 1
            self.logger.info(f"Debug attempt {attempts}/{self.cfg.max_debug_attempts} for node {current.id}")
            debug_node = self._debug(current)
            state, eval_result = self._evaluate_and_record(task, state, debug_node)
            current = debug_node
        return state, eval_result

    def _eligible_program_ids(self) -> set[int]:
        ids = set()
        for node in self.journal.good_nodes:
            if isinstance(node.metric, MetricValue) and node.metric.info and "active_program_id" in node.metric.info:
                ids.add(int(node.metric.info["active_program_id"]))
        return ids

    def _node_for_program_id(self, program_id: int) -> Node | None:
        for node in self.journal.good_nodes:
            if (
                isinstance(node.metric, MetricValue)
                and node.metric.info
                and int(node.metric.info.get("active_program_id", -1)) == int(program_id)
            ):
                return node
        return None

    def _can_continue(self, task) -> bool:
        if self.state.running_time >= self.cfg.time_limit_secs:
            return False
        if self.state.current_step >= self.cfg.step_limit:
            return False
        if self._has_program_budget(task):
            return True
        return bool(self.cfg.query_when_program_budget_exhausted and task.label_budget_remaining() > 0)

    def _has_program_budget(self, task) -> bool:
        program_budget = int(getattr(task.cfg, "program_budget", self.cfg.step_limit))
        return task.valid_program_count() < program_budget and self.state.current_step < self.cfg.step_limit

    def _log_state(self) -> None:
        best_node = self.journal.get_best_node()
        best_node_step = 0 if best_node is None else best_node.step
        if self.journal.nodes:
            self.logger.log(
                self.journal.get_node_data(len(self.journal.nodes) - 1) | {"current_best_node": best_node_step},
                "JOURNAL",
                step=max(0, self.state.current_step),
            )
        self.logger.log(self.state.state_dict(), "STATE", step=max(0, self.state.current_step))

    def _record_controller_event(self, task, event: dict[str, Any]) -> None:
        event = {
            "controller_iter": self.controller_iteration,
            "solver_step": self.state.current_step,
            **event,
        }
        if hasattr(task, "record_controller_event"):
            task.record_controller_event(event)
        self.logger.info(f"Controller event: {json.dumps(event, default=str)}")

    def _serializable_action_info(self, action_info: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in action_info.items() if key != "target_node"}

    def parse_eval_result(self, node: Node, eval_result: dict[str, Any]):
        super().parse_eval_result(node, eval_result)
        if node.is_buggy and not eval_result.get(VALID_SOLUTION, True):
            info = node.metric.info if node.metric else {}
            node.metric = WorstMetricValue(info=info)

    def save_checkpoint(self):
        super().save_checkpoint()
        path = Path(self.cfg.checkpoint_path) / "coda_active_search_state.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"controller_iteration": self.controller_iteration}, f, indent=2)

    def load_checkpoint(self):
        super().load_checkpoint()
        path = Path(self.cfg.checkpoint_path) / "coda_active_search_state.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            self.controller_iteration = int(data.get("controller_iteration", 0))
