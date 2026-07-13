from dataclasses import dataclass, field
from typing import Any

from omegaconf import SI

from dojo.config_dataclasses.task.agentssl import AgentSSLTaskConfig


@dataclass
class ActiveProgramSelectionTaskConfig(AgentSSLTaskConfig):
    setting: str = field(
        default="aSSL",
        metadata={
            "help": "Active program-selection setting.",
        },
    )
    query_split: str = field(
        default="query",
        metadata={
            "help": "Name of the unlabeled pool used for active label queries.",
        },
    )
    oracle_dir: str = field(
        default=SI("${task.cache_dir}/${task.name}/oracle"),
        metadata={
            "help": "Host-only directory containing hidden labels for active queries.",
            "exclude_from_hash": True,
        },
    )
    query_ann_file: str = field(
        default=SI("${task.data_dir}/annotations/${task.query_split}/${task.query_split}.json"),
        metadata={
            "help": "Visible unlabeled query annotation file mounted into the container.",
            "exclude_from_hash": True,
        },
    )
    oracle_query_ann_file: str = field(
        default=SI("${task.oracle_dir}/annotations/${task.query_split}/${task.query_split}.json"),
        metadata={
            "help": "Host-only labeled query annotation file used by QueryLabel.",
            "exclude_from_hash": True,
        },
    )
    program_budget: int = field(
        default=20,
        metadata={
            "help": "Maximum number of programs the agent may generate/evaluate.",
        },
    )
    label_budget: int = field(
        default=50,
        metadata={
            "help": "Maximum number of ground-truth query labels the agent may request.",
        },
    )
    selection_method: str = field(
        default="coda",
        metadata={
            "help": "Host-side active program-selection method.",
        },
    )
    coda_alpha: float = field(
        default=0.9,
        metadata={
            "help": "CODA alpha. CODA uses prior_strength = 1 - alpha, so lower values increase consensus-prior weight.",
        },
    )
    coda_learning_rate: float = field(
        default=0.01,
        metadata={
            "help": "CODA label-update strength.",
        },
    )
    active_policy: dict[str, Any] = field(
        default_factory=lambda: {
            "name": "disabled",
            "query_every_n_programs": 0,
            "start_after_programs": 0,
            "queries_per_interval": 1,
            "max_queries_per_step": 1,
        },
        metadata={
            "help": "Host-side active label-query policy. Disabled by default.",
        },
    )

    def validate(self) -> None:
        super().validate()
        if self.program_budget <= 0:
            raise ValueError("program_budget must be positive.")
        if self.label_budget < 0:
            raise ValueError("label_budget must be non-negative.")
        if not 0.0 <= self.coda_alpha <= 1.0:
            raise ValueError("coda_alpha must be in [0, 1].")
        if self.coda_learning_rate < 0.0:
            raise ValueError("coda_learning_rate must be non-negative.")
        if self.query_ann_file == self.oracle_query_ann_file:
            raise ValueError("query_ann_file and oracle_query_ann_file must be different.")
