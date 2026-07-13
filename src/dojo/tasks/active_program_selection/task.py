from __future__ import annotations

import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from dojo.config_dataclasses.task.active_program_selection import (
    ActiveProgramSelectionTaskConfig,
)
from dojo.core.interpreters.base import ExecutionResult, Interpreter
from dojo.core.solvers.utils.metric import MetricValue, WorstMetricValue
from dojo.core.tasks.constants import (
    AUX_EVAL_INFO,
    EXECUTION_OUTPUT,
    TASK_DESCRIPTION,
    TEST_FITNESS,
    VALID_SOLUTION,
    VALID_SOLUTION_FEEDBACK,
    VALIDATION_FITNESS,
    WARM_START_PROGRAM,
)
from dojo.tasks.agentssl.task import AgentSSLTask
from dojo.tasks.active_program_selection.active_policies import active_policy_num_queries
from dojo.utils.code_parsing import extract_code, format_code, write_code_to_file
from dojo.utils.output_parsing import extract_metrics


def _ensure_agentssl_importable() -> None:
    for parent in Path(__file__).resolve().parents:
        if (parent / "agentssl" / "selection").is_dir():
            path = str(parent)
            if path not in sys.path:
                sys.path.insert(0, path)
            return


_ensure_agentssl_importable()

from agentssl.selection.coda import CODA  # noqa: E402


@dataclass
class ProgramRecord:
    program_id: int
    node_id: str | None
    artifact_dir: str
    probs_path: str
    valid: bool


class _InMemoryCODADataset:
    def __init__(self, preds: torch.Tensor):
        self.preds = preds.float()
        self.device = self.preds.device
        self.labels = None


class ActiveProgramSelectionTask(AgentSSLTask):
    def __init__(self, cfg: ActiveProgramSelectionTaskConfig) -> None:
        super().__init__(cfg)
        self.cfg: ActiveProgramSelectionTaskConfig = cfg
        self.warm_start_program = Path(self.task_dir).resolve() / "warm_start_program.py"
        self.workspace_dir = self.experiment_dir / "workspace"
        self.selection_dir = self.experiment_dir / "active_program_selection"
        self.programs_dir = self.selection_dir / "programs"
        self.stacked_probs_path = self.selection_dir / "query_probs.pt"
        self.program_records_path = self.selection_dir / "program_records.json"
        self.queried_labels_path = self.selection_dir / "queried_labels.json"
        self.coda_state_path = self.selection_dir / "coda_state.json"
        self.coda_pbest_history_path = self.selection_dir / "coda_pbest_history.pt"
        self.oracle_query_ann_file = Path(self.cfg.oracle_query_ann_file).resolve()
        self.query_ann_file = Path(self.cfg.query_ann_file).resolve()

        self.selection_dir.mkdir(parents=True, exist_ok=True)
        self.programs_dir.mkdir(parents=True, exist_ok=True)

        self.program_records: list[ProgramRecord] = self._load_program_records()
        self.queried_labels: dict[int, dict[str, Any]] = self._load_queried_labels()
        self._latest_unbound_program_id: int | None = None
        self._last_coda = None
        self._last_pbest: list[float] = []
        self._last_program_ids: list[int] = []

    def prepare(self, **task_args):
        state = task_args
        state["init_obs"] = {
            "program_budget": self.cfg.program_budget,
            "label_budget": self.cfg.label_budget,
            "selection_method": self.cfg.selection_method,
        }
        if not self.warm_start_program.exists():
            warm_start_program = ""
        else:
            warm_start_program = self.warm_start_program.read_text()
        task_info = {
            TASK_DESCRIPTION: self.task_description,
            "lower_is_better": False,
            WARM_START_PROGRAM: warm_start_program,
        }

        return state, task_info

    def step_task(self, state: Dict[str, Any], action: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        try:
            solution = extract_code(action)
        except Exception as e:
            self.logger.error(f"The solution does not follow the required format: {e}")
            exec_output = ExecutionResult.get_empty()
            exec_output.term_out[0] = f"Invalid solution: {e}"
            return state, {EXECUTION_OUTPUT: exec_output, VALIDATION_FITNESS: None, VALID_SOLUTION: False}

        write_code_to_file(solution, self.write_program_path)
        stale_outputs = self.workspace_dir / "query_outputs"
        if stale_outputs.exists():
            shutil.rmtree(stale_outputs)
        executable = format_code(self.eval_script)

        interpreter = state["solver_interpreter"]
        exec_output: ExecutionResult = interpreter.run(executable)
        eval_result = {EXECUTION_OUTPUT: exec_output}

        if (not exec_output.exit_code == 0) or exec_output.timed_out:
            self.logger.error(
                "Execution failed - "
                f"exit code: {exec_output.exit_code} - "
                f"timed out: {exec_output.timed_out} - "
                f"execution time: {exec_output.exec_time}"
            )
            eval_result[VALID_SOLUTION] = False
            return state, eval_result

        metrics = extract_metrics(exec_output.term_out)
        self.logger.info(f"Extracted active eval metadata: {metrics}")
        if not metrics:
            eval_result[VALID_SOLUTION] = False
            return state, eval_result

        try:
            program_record = self._record_program_outputs(solution)
            self._latest_unbound_program_id = program_record.program_id
            coda_state = self.rerun_coda(history_event="program_evaluated")
            fitness = self._fitness_for_program(program_record.program_id, coda_state)
        except Exception as e:
            self.logger.exception(f"Active program-selection bookkeeping failed: {e}")
            exec_output.term_out.append(f"\nActive bookkeeping failed: {e}\n")
            eval_result[VALID_SOLUTION] = False
            return state, eval_result

        aux_info = {
            **metrics,
            "active_program_id": program_record.program_id,
            "active_program_artifact_dir": program_record.artifact_dir,
            "active_selection": coda_state,
            "score": fitness,
        }

        eval_result[VALID_SOLUTION] = True
        eval_result[VALID_SOLUTION_FEEDBACK] = "Program evaluated and query predictions saved"
        eval_result[VALIDATION_FITNESS] = fitness
        eval_result[TEST_FITNESS] = fitness
        eval_result[AUX_EVAL_INFO] = aux_info
        return state, eval_result

    def query_label(self, item_idx: int | None = None) -> dict[str, Any]:
        if len(self.queried_labels) >= self.cfg.label_budget:
            raise RuntimeError("Label budget exhausted.")

        coda_state = self.rerun_coda(history_event="label_selection")
        selector = self._last_coda
        if selector is None:
            raise RuntimeError("No evaluated non-buggy programs are available for CODA.")

        query_score = None
        if item_idx is None:
            item_idx, query_score = selector.get_next_item_to_label()
            item_idx = int(item_idx)
        else:
            item_idx = int(item_idx)

        if item_idx in self.queried_labels:
            raise ValueError(f"Query item {item_idx} has already been labeled.")

        oracle = self._oracle_labels()
        if item_idx < 0 or item_idx >= len(oracle["items"]):
            raise IndexError(f"Query item {item_idx} is outside the query pool.")

        item = oracle["items"][item_idx]
        record = {
            "item_idx": item_idx,
            "image_id": item["image_id"],
            "file_name": item["file_name"],
            "category_id": item["category_id"],
            "class_idx": item["class_idx"],
            "query_score": query_score,
        }
        self.queried_labels[item_idx] = record
        self._save_json(
            self.queried_labels_path,
            {str(k): v for k, v in sorted(self.queried_labels.items())},
        )
        coda_state = self.rerun_coda(history_event="label_added")
        return {
            **record,
            "label_budget_used": len(self.queried_labels),
            "label_budget_remaining": self.cfg.label_budget - len(self.queried_labels),
            "active_selection": coda_state,
        }

    def rerun_coda(
        self,
        allowed_program_ids: set[int] | None = None,
        history_event: str = "coda_rerun",
        record_history: bool = True,
    ) -> dict[str, Any]:
        probs, program_ids = self._load_stacked_probs(allowed_program_ids=allowed_program_ids)
        if probs is None:
            self._last_coda = None
            self._last_pbest = []
            self._last_program_ids = []
            state = {
                "method": self.cfg.selection_method,
                "num_programs": 0,
                "num_query_items": 0,
                "num_labels": len(self.queried_labels),
                "programs": [],
            }
            self._save_json(self.coda_state_path, state)
            return state

        if self.cfg.selection_method != "coda":
            raise NotImplementedError(f"Unsupported active selection method: {self.cfg.selection_method}")

        dataset = _InMemoryCODADataset(probs)
        selector = CODA(
            dataset,
            alpha=self.cfg.coda_alpha,
            learning_rate=self.cfg.coda_learning_rate,
        )

        replayed_labels = []
        for item_idx, label in sorted(self.queried_labels.items()):
            if item_idx in selector.unlabeled_idxs:
                selector.add_label(item_idx, int(label["class_idx"]), float(label.get("query_score") or 0.0))
                replayed_labels.append(item_idx)

        pbest_tensor = selector.get_pbest().detach().cpu().reshape(-1)
        pbest = [float(x) for x in pbest_tensor.tolist()]
        best_local_idx = int(torch.argmax(pbest_tensor).item())
        best_program_id = int(program_ids[best_local_idx])

        rows = [
            {
                "program_id": int(program_id),
                "pbest": float(pbest[idx]),
                "rank": 1 + sum(other > pbest[idx] for other in pbest),
            }
            for idx, program_id in enumerate(program_ids)
        ]
        rows.sort(key=lambda r: (-r["pbest"], r["program_id"]))

        state = {
            "method": "coda",
            "coda_alpha": float(self.cfg.coda_alpha),
            "coda_prior_strength": float(1.0 - self.cfg.coda_alpha),
            "coda_learning_rate": float(self.cfg.coda_learning_rate),
            "num_programs": len(program_ids),
            "num_query_items": int(probs.shape[1]),
            "num_classes": int(probs.shape[2]),
            "num_labels": len(self.queried_labels),
            "replayed_label_item_idxs": replayed_labels,
            "best_program_id": best_program_id,
            "programs": rows,
        }
        self._last_coda = selector
        self._last_pbest = pbest
        self._last_program_ids = [int(x) for x in program_ids]
        self._save_json(self.coda_state_path, state)
        if record_history:
            self._append_coda_pbest_history(state, history_event)
        return state

    def refresh_node_fitness(self, journal) -> None:
        self._bind_latest_program_to_journal(journal)
        allowed_ids = {
            int(node.metric.info["active_program_id"])
            for node in journal.nodes
            if not node.is_buggy
            and isinstance(node.metric, MetricValue)
            and node.metric.info
            and "active_program_id" in node.metric.info
        }
        coda_state = self.rerun_coda(allowed_program_ids=allowed_ids, record_history=False)
        pbest_by_program_id = {
            int(row["program_id"]): float(row["pbest"])
            for row in coda_state.get("programs", [])
        }
        for node in journal.nodes:
            if (
                node.is_buggy
                or not isinstance(node.metric, MetricValue)
                or not node.metric.info
                or "active_program_id" not in node.metric.info
            ):
                continue
            program_id = int(node.metric.info["active_program_id"])
            if program_id not in pbest_by_program_id:
                node.metric = WorstMetricValue(info=node.metric.info)
                node.is_buggy = True
                continue
            info = {
                **node.metric.info,
                "active_selection": coda_state,
                "pbest": pbest_by_program_id[program_id],
                "score": pbest_by_program_id[program_id],
            }
            node.metric = MetricValue(pbest_by_program_id[program_id], maximize=True, info=info)

    def after_program_evaluated(self, journal) -> list[dict[str, Any]]:
        self.refresh_node_fitness(journal)
        events = self._apply_active_policy()
        if events:
            self.refresh_node_fitness(journal)
        return events

    def _apply_active_policy(self) -> list[dict[str, Any]]:
        events = []
        for _ in range(active_policy_num_queries(self)):
            if len(self.queried_labels) >= self.cfg.label_budget:
                break
            events.append(self.query_label())
        return events

    def active_policy(self) -> dict[str, Any]:
        policy = getattr(self.cfg, "active_policy", None)
        if policy is None:
            return {"name": "disabled"}
        if hasattr(policy, "items"):
            return dict(policy.items())
        return dict(policy)

    def _record_program_outputs(self, solution: str) -> ProgramRecord:
        program_id = len(self.program_records)
        output_dir = self.workspace_dir / "query_outputs"
        probs_path = output_dir / "query_probs.pt"
        if not probs_path.exists():
            raise FileNotFoundError(f"Expected query probabilities at {probs_path}")

        artifact_dir = self.programs_dir / f"program_{program_id:04d}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        write_code_to_file(solution, artifact_dir / "program.py")

        for src in output_dir.iterdir():
            if src.is_file():
                shutil.copy2(src, artifact_dir / src.name)

        program_probs_path = artifact_dir / "query_probs.pt"
        self._append_probs(program_probs_path)

        record = ProgramRecord(
            program_id=program_id,
            node_id=None,
            artifact_dir=str(artifact_dir),
            probs_path=str(program_probs_path),
            valid=True,
        )
        self.program_records.append(record)
        self._save_program_records()
        return record

    def _append_probs(self, probs_path: Path) -> None:
        new_probs = torch.load(probs_path, map_location="cpu").float()
        if new_probs.ndim != 2:
            raise ValueError(f"Expected query_probs shape (N, C), got {tuple(new_probs.shape)}")

        if self.stacked_probs_path.exists():
            old = torch.load(self.stacked_probs_path, map_location="cpu").float()
            if old.ndim != 3:
                raise ValueError(f"Expected stacked query_probs shape (H, N, C), got {tuple(old.shape)}")
            if old.shape[1:] != new_probs.shape:
                raise ValueError(
                    "New query_probs shape does not match previous programs: "
                    f"old={tuple(old.shape)}, new={tuple(new_probs.shape)}"
                )
            stacked = torch.cat([old, new_probs.unsqueeze(0)], dim=0)
        else:
            stacked = new_probs.unsqueeze(0)

        torch.save(stacked, self.stacked_probs_path)

    def _load_stacked_probs(self, allowed_program_ids: set[int] | None = None):
        if not self.stacked_probs_path.exists() or not self.program_records:
            return None, []

        stacked = torch.load(self.stacked_probs_path, map_location="cpu").float()
        keep_indices = []
        keep_program_ids = []
        for idx, record in enumerate(self.program_records):
            if not record.valid:
                continue
            if allowed_program_ids is not None and record.program_id not in allowed_program_ids:
                continue
            keep_indices.append(idx)
            keep_program_ids.append(record.program_id)

        if not keep_indices:
            return None, []

        return stacked[keep_indices], keep_program_ids

    def _fitness_for_program(self, program_id: int, coda_state: dict[str, Any]) -> float:
        for row in coda_state.get("programs", []):
            if int(row["program_id"]) == int(program_id):
                return float(row["pbest"])
        return 0.0

    def _append_coda_pbest_history(self, state: dict[str, Any], event: str) -> None:
        programs = state.get("programs", [])
        if not programs:
            return

        program_ids = [int(row["program_id"]) for row in programs]
        pbest = [float(row["pbest"]) for row in programs]
        selected_program_id = int(program_ids[0])

        row_ids = torch.tensor(program_ids, dtype=torch.long)
        row_pbest = torch.tensor(pbest, dtype=torch.float32)
        row_count = torch.tensor([len(program_ids)], dtype=torch.long)
        row_selected = torch.tensor([selected_program_id], dtype=torch.long)
        row_num_labels = torch.tensor([int(state.get("num_labels", 0))], dtype=torch.long)

        if self.coda_pbest_history_path.exists():
            history = torch.load(self.coda_pbest_history_path, map_location="cpu")
            pbest_hist = history["pbest"]
            ids_hist = history["program_ids"]
            max_width = max(int(pbest_hist.shape[1]), len(program_ids))
            if max_width > pbest_hist.shape[1]:
                pbest_hist = self._pad_columns(pbest_hist, max_width, float("nan"))
                ids_hist = self._pad_columns(ids_hist, max_width, -1)
            row_pbest = self._pad_columns(row_pbest.unsqueeze(0), max_width, float("nan")).squeeze(0)
            row_ids = self._pad_columns(row_ids.unsqueeze(0), max_width, -1).squeeze(0)
            history = {
                **history,
                "pbest": torch.cat([pbest_hist, row_pbest.unsqueeze(0)], dim=0),
                "program_ids": torch.cat([ids_hist, row_ids.unsqueeze(0)], dim=0),
                "num_programs": torch.cat([history["num_programs"], row_count], dim=0),
                "selected_program_id": torch.cat([history["selected_program_id"], row_selected], dim=0),
                "num_labels": torch.cat([history["num_labels"], row_num_labels], dim=0),
                "event": [*history.get("event", []), event],
            }
        else:
            history = {
                "pbest": row_pbest.unsqueeze(0),
                "program_ids": row_ids.unsqueeze(0),
                "num_programs": row_count,
                "selected_program_id": row_selected,
                "num_labels": row_num_labels,
                "event": [event],
            }

        torch.save(history, self.coda_pbest_history_path)

    @staticmethod
    def _pad_columns(tensor: torch.Tensor, width: int, value: float | int) -> torch.Tensor:
        if tensor.shape[1] >= width:
            return tensor
        pad = torch.full(
            (tensor.shape[0], width - tensor.shape[1]),
            value,
            dtype=tensor.dtype,
        )
        return torch.cat([tensor, pad], dim=1)

    def _bind_latest_program_to_journal(self, journal) -> None:
        if self._latest_unbound_program_id is None:
            return
        for node in reversed(journal.nodes):
            if (
                isinstance(node.metric, MetricValue)
                and node.metric.info
                and node.metric.info.get("active_program_id") == self._latest_unbound_program_id
            ):
                record = self.program_records[self._latest_unbound_program_id]
                if record.node_id is None:
                    record.node_id = node.id
                    self._save_program_records()
                self._latest_unbound_program_id = None
                return

    def _oracle_labels(self) -> dict[str, Any]:
        with open(self.oracle_query_ann_file) as f:
            coco = json.load(f)
        with open(self.task_dir / "annotations" / "train" / "train.json") as f:
            train = json.load(f)

        category_to_class = {
            int(category["id"]): idx
            for idx, category in enumerate(train["categories"])
        }
        image_id_to_name = {
            int(image["id"]): Path(image["file_name"]).name
            for image in coco["images"]
        }
        items = []
        for ann in sorted(
            coco["annotations"],
            key=lambda a: image_id_to_name[int(a["image_id"])],
        ):
            image_id = int(ann["image_id"])
            category_id = int(ann["category_id"])
            items.append(
                {
                    "image_id": image_id,
                    "file_name": image_id_to_name[image_id],
                    "category_id": category_id,
                    "class_idx": category_to_class[category_id],
                }
            )
        return {"items": items}

    def _load_program_records(self) -> list[ProgramRecord]:
        if not self.program_records_path.exists():
            return []
        with open(self.program_records_path) as f:
            return [ProgramRecord(**record) for record in json.load(f)]

    def _save_program_records(self) -> None:
        self._save_json(self.program_records_path, [asdict(record) for record in self.program_records])

    def _load_queried_labels(self) -> dict[int, dict[str, Any]]:
        if not self.queried_labels_path.exists():
            return {}
        with open(self.queried_labels_path) as f:
            return {int(k): v for k, v in json.load(f).items()}

    @staticmethod
    def _save_json(path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def evaluate_fitness(
        self,
        solution: Optional[Any] = None,
        state: Optional[Dict[str, Any]] = None,
        interpreter: Optional[Interpreter] = None,
        aux_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        _, eval_result = self.step_task(state, solution)
        return eval_result
