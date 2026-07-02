from __future__ import annotations
from pathlib import Path
from dojo.config_dataclasses.task.agentssl import AgentSSLTaskConfig


def vtab_binds(task: AgentSSLTaskConfig) -> tuple[dict[str, str], dict[str, str]]:
    experiment_dir = Path(task.experiment_dir).resolve()
    workspace_dir = experiment_dir / "workspace"
    program_path = experiment_dir / "program.py"

    experiment_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    if not program_path.exists():
        program_path.write_text("# placeholder — agent solution will be written here\n")

    ro_binds = {
        str(Path(task.task_dir).resolve()): "/task",
        str(Path(task.eval_dir).resolve()): "/eval",
        str(Path(task.ssl_dir).resolve() / task.benchmark / "eval" / "shared" / "assets"): "/assets",
        str(Path(task.ssl_dir).resolve() / task.benchmark / "eval" / "shared" / "core"): "/eval/src",
        str(program_path): "/program.py",
    }

    rw_binds = {
        str(workspace_dir): "/workspace",
    }

    return ro_binds, rw_binds
