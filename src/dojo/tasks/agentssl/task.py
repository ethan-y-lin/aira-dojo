# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
import json
from dojo.core.interpreters.base import ExecutionResult, Interpreter
from dojo.core.tasks.base import Task
from dojo.core.tasks.constants import (
    EXECUTION_OUTPUT,
    TASK_DESCRIPTION,
    WARM_START_PROGRAM,
    TEST_FITNESS,
    VALID_SOLUTION_FEEDBACK,
    VALIDATION_FITNESS,
    AUX_EVAL_INFO,
    VALID_SOLUTION,
)
from dojo.utils.code_parsing import extract_code, format_code, write_code_to_file
from dojo.utils.output_parsing import extract_metrics
from dojo.config_dataclasses.task.agentssl import AgentSSLTaskConfig

def validate_submission(submission: Path) -> tuple[bool, str]:
    """
    Validates a submission for the given competition by actually running the competition grader.
    This is designed for end users, not developers (we assume that the competition grader is
    correctly implemented and use that for validating the submission, not the other way around).
    """
    if not submission.is_file():
        return False, f"Submission invalid! Submission file {submission} does not exist."

    if not submission.suffix.lower() == ".json":
        return False, "Submission invalid! Submission file must be a JSON file."

    return True, "Submission is valid."

def parse_report(report: Dict[str, Any]):
    parsed_report = {}
    for key, value in report.items():
        if isinstance(value, (bool, int, float)):
            parsed_report[key] = float(value)
        else:
            parsed_report[key] = value
    return parsed_report

class AgentSSLTask(Task):
    """
    Represents an AgentSSL task.

    This task executes a provided solution to train a model and expects the solution
    to produce a submission file (submission.csv). It then runs an evaluation script to
    compute a fitness score.

    Expected configuration keys (accessible via self.cfg):
      - eval_script: The evaluation script code.
      - task_description: The task description.
    """

    def __init__(self, cfg: AgentSSLTaskConfig) -> None:
        """
        Initialize the AgentSSLTask.

        Args:
            **cfg: Configuration containing keys like 'eval_script' and 'task_description'.
        """
        super().__init__(cfg)

        # Get instructions
        self.task_src_path = Path(__file__).resolve().parent
        self.instructions_path = self.task_src_path / "instructions.txt"
        self.instructions = self.instructions_path.read_text()
        self.instructions = os.path.expandvars(self.instructions)
        self.shot = self.cfg.shot
        self.task_dir = Path(self.cfg.task_dir).resolve()
        self.ssl_dir = Path(self.cfg.ssl_dir).resolve()
        self.eval_dir = Path(self.cfg.eval_dir).resolve()
        self.warm_start_program = Path(self.task_dir).resolve() / "warm_start_program.py"
        
        # Read task description.
        task_description_path = Path(self.task_dir).resolve() / "description.md"
        self.task_description = self.instructions + "\n" + task_description_path.read_text()
        eval_script_path = Path(self.eval_dir).resolve() / "evaluate.py"
        self.eval_script = eval_script_path.read_text()

        # Experiment directory
        self.experiment_dir = Path(self.cfg.experiment_dir).resolve()
        self.write_program_path = self.experiment_dir / "program.py"

    def prepare(self, **task_args):
        state = task_args
        state["init_obs"] = {}
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
        """
        Execute a single step of the task.

        In this task, a step corresponds to evaluating the provided solution code.

        Args:
            state (Dict[str, Any]): The current state of the task.
            action (Any): The solution code (as a string) to evaluate.

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: A tuple containing the updated state and the outcome.
        """
        try:
            solution = extract_code(action)
        except Exception as e:
            self.logger.error(f"The solution does not follow the required format: {e}")
            exec_output = ExecutionResult.get_empty()
            exec_output.term_out[0] = f"Invalid solution: {e}"
            return state, {EXECUTION_OUTPUT: exec_output, VALIDATION_FITNESS: None, VALID_SOLUTION: False}

        # Write the program (solution) to the experiment directory
        write_code_to_file(solution, self.write_program_path)
        # Format the evaluation script (which should access the program.py file)
        executable = format_code(self.eval_script)
        # Execute the evaluation script
        interpreter = state["solver_interpreter"]
        exec_output: ExecutionResult = interpreter.run(executable)
        eval_result = {EXECUTION_OUTPUT: exec_output}
        # Check if the execution was not successful
        if (not exec_output.exit_code == 0) or exec_output.timed_out:
            self.logger.error(f"Execution failed - exit code: {exec_output.exit_code} - timed out: {exec_output.timed_out} - execution time: {exec_output.exec_time}")
        else: 
            self.logger.info(f"Execution successful.")      
            metrics = extract_metrics(exec_output.term_out)
            self.logger.info(f"Extracted Metrics: {metrics}")
            if metrics:
                eval_result[VALID_SOLUTION] = True
                eval_result[VALID_SOLUTION_FEEDBACK] = "Solution is valid"
                eval_result[TEST_FITNESS] =  metrics["fitness"]
                eval_result[AUX_EVAL_INFO] = metrics
                eval_result[AUX_EVAL_INFO]["score"] = eval_result[TEST_FITNESS]
            else:
                eval_result[VALID_SOLUTION] = False

        return state, eval_result

    def evaluate_fitness(
        self,
        solution: Optional[Any] = None,
        state: Optional[Dict[str, Any]] = None,
        interpreter: Optional[Interpreter] = None,
        aux_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        _, eval_result = self.step_task(state, solution)
        return eval_result

    def close(self, state):
        for interp_key in ["solver_interpreter", "eval_interpreter"]:
            if interp_key not in state:
                continue

            interpreter = state[interp_key]
            if hasattr(interpreter, "cleanup_session"):
                interpreter.cleanup_session()

            if hasattr(interpreter, "clean_up"):
                interpreter.clean_up()

