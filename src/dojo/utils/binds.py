from __future__ import annotations
from pathlib import Path
from dojo.config_dataclasses.task.agentssl import AgentSSLTaskConfig


def vtab_binds(task: AgentSSLTaskConfig) -> tuple[dict[str, str], dict[str, str]]:
    experiment_dir = Path(task.experiment_dir).resolve()
    workspace_dir = experiment_dir / "workspace"
    program_path = experiment_dir / "program.py"
    data_dir = Path(task.data_dir).resolve()
    dataset_root = data_dir.parent
    query_annotations_file = dataset_root / "annotations" / "query" / "query.json"
    if not query_annotations_file.exists():
        query_annotations_file = data_dir / "annotations" / "query" / "query.json"

    experiment_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    if not program_path.exists():
        program_path.write_text("# placeholder — agent solution will be written here\n")

    ro_binds = {
        str(data_dir / "images" / "train"): "/data/images/train",
        str(data_dir / "images" / "val"): "/data/images/val",
        str(query_annotations_file): "/data/annotations/val/val.json",
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
