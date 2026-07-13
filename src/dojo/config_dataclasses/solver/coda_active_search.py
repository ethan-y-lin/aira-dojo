from dataclasses import dataclass, field

from dojo.config_dataclasses.solver.greedy import GreedySolverConfig


@dataclass
class CODAActiveSearchSolverConfig(GreedySolverConfig):
    initial_drafts: int = field(
        default=5,
        metadata={
            "description": "Number of valid initial draft programs to collect before using the CODA controller.",
            "example": 5,
        },
    )
    uncertainty_threshold: float = field(
        default=0.6,
        metadata={
            "description": "Tune when normalized CODA P_best entropy is below this value.",
            "example": 0.6,
        },
    )
    diversity_threshold: float = field(
        default=0.05,
        metadata={
            "description": "Draft when uncertainty is high and average pairwise JSD is below this value.",
            "example": 0.05,
        },
    )
    tune_temperature: float = field(
        default=0.5,
        metadata={
            "description": "Temperature for sharpened CODA P_best tune-target sampling. Must be positive.",
            "example": 0.5,
        },
    )
    max_debug_attempts: int = field(
        default=3,
        metadata={
            "description": "Maximum number of immediate debug attempts for each invalid generated program.",
            "example": 3,
        },
    )
    query_when_program_budget_exhausted: bool = field(
        default=False,
        metadata={
            "description": "Whether to keep spending labels after the program budget is exhausted.",
            "example": False,
        },
    )

    def validate(self) -> None:
        super().validate()
        if self.initial_drafts <= 0:
            raise ValueError("initial_drafts must be positive.")
        if not 0.0 <= self.uncertainty_threshold <= 1.0:
            raise ValueError("uncertainty_threshold must be in [0, 1].")
        if self.diversity_threshold < 0.0:
            raise ValueError("diversity_threshold must be non-negative.")
        if self.tune_temperature <= 0.0:
            raise ValueError("tune_temperature must be positive.")
        if self.max_debug_attempts < 0:
            raise ValueError("max_debug_attempts must be non-negative.")
