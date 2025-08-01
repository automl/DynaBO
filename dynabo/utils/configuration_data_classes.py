"""Configuration data classes for experiment settings."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Optional


class PriorKind(str, Enum):
    GOOD = "good"
    MEDIUM = "medium"
    MISLEADING = "misleading"
    DECEIVING = "deceiving"


class ValidationMethod(str, Enum):
    MANN_WHITNEY_U = "mann_whitney_u"
    DIFFERENCE = "difference"


@dataclass
class BenchmarkConfig:
    benchmarklib: Literal["yahpogym", "mfpbench"]
    scenario: str
    dataset: str
    metric: str

    def __post_init__(self):
        if self.benchmarklib not in ["yahpogym", "mfpbench"]:
            raise ValueError(f"Unsupported benchmarklib: {self.benchmarklib}")

    @classmethod
    def from_config(cls, config: dict) -> "BenchmarkConfig":
        """Extract benchmark related configuration."""
        return cls(benchmarklib=config["benchmarklib"], scenario=config["scenario"], dataset=config["dataset"], metric=config["metric"])


@dataclass
class SMACConfig:
    timeout: int
    seed: int
    n_trials: int

    def __post_init__(self):
        if self.timeout <= 0:
            raise ValueError(f"Timeout must be positive, got {self.timeout}")
        if self.n_trials <= 0:
            raise ValueError(f"Number of trials must be positive, got {self.n_trials}")

    @classmethod
    def from_config(cls, config: dict) -> "SMACConfig":
        """Extract SMAC base configuration."""
        return cls(timeout=int(config["timeout_total"]), seed=int(config["seed"]), n_trials=int(config["n_trials"]))


@dataclass
class InitialDesignConfig:
    n_configs_per_hyperparameter: int
    max_ratio: float = field(default=0.25)

    def __post_init__(self):
        if self.n_configs_per_hyperparameter <= 0:
            raise ValueError(f"Configs per hyperparameter must be positive, got {self.n_configs_per_hyperparameter}")
        if not 0 < self.max_ratio <= 1:
            raise ValueError(f"Max ratio must be between 0 and 1, got {self.max_ratio}")

    @classmethod
    def from_config(cls, config: dict) -> "InitialDesignConfig":
        """Extract initial design configuration."""
        return cls(n_configs_per_hyperparameter=int(config["initial_design__n_configs_per_hyperparameter"]), max_ratio=float(config["initial_design__max_ratio"]))


@dataclass
class PriorConfig:
    kind: PriorKind
    chance_theta: float
    std_denominator: float = field(default=5.0)
    no_incumbent_percentile: float = field(default=50.0)

    def __post_init__(self):
        if isinstance(self.kind, str):
            try:
                self.kind = PriorKind(self.kind)
            except ValueError:
                raise ValueError(f"Invalid prior kind: {self.kind}")
        if self.chance_theta is not None and not 0 <= self.chance_theta <= 1:
            raise ValueError(f"Chance theta must be between 0 and 1, got {self.chance_theta}")
        if self.std_denominator is not None and self.std_denominator <= 0:
            raise ValueError(f"Std denominator must be positive, got {self.std_denominator}")
        if not 0 <= self.no_incumbent_percentile <= 100:
            raise ValueError(f"No incumbent percentile must be between 0 and 100, got {self.no_incumbent_percentile}")

    @classmethod
    def from_config(cls, config: dict) -> "PriorConfig":
        """Extract basic prior configuration."""
        return cls(
            kind=config["prior_kind"],
            chance_theta=float(config["prior_chance_theta"]) if config["prior_chance_theta"] is not None else None,
            std_denominator=float(config["prior_std_denominator"]),
            no_incumbent_percentile=float(config["no_incumbent_percentile"]),
        )


@dataclass
class PriorDecayConfig:
    enumerator: float = field(default=200.0)
    denominator: float = field(default=10.0)

    def __post_init__(self):
        if self.enumerator <= 0:
            raise ValueError(f"Decay enumerator must be positive, got {self.enumerator}")
        if self.denominator <= 0:
            raise ValueError(f"Decay denominator must be positive, got {self.denominator}")

    @classmethod
    def from_config(cls, config: dict) -> "PriorDecayConfig":
        """Extract prior decay configuration."""
        return cls(enumerator=float(config["prior_decay_enumerator"]), denominator=float(config["prior_decay_denominator"]))


@dataclass
class PriorValidationConfig:
    validate: bool = field(default=True)
    method: ValidationMethod = field(default=ValidationMethod.MANN_WHITNEY_U)
    n_samples: Optional[int] = field(default=500)
    manwhitney_p_value: Optional[float] = field(default=0.05)
    difference_threshold: Optional[float] = field(default=-1.0)

    def __post_init__(self):
        if isinstance(self.method, str):
            try:
                self.method = ValidationMethod(self.method)
            except ValueError:
                raise ValueError(f"Invalid validation method: {self.method}")
        if self.n_samples is not None and self.n_samples <= 0:
            raise ValueError(f"Number of samples must be positive, got {self.n_samples}")
        if self.manwhitney_p_value is not None and not 0 < self.manwhitney_p_value < 1:
            raise ValueError(f"Mann-Whitney p-value must be between 0 and 1, got {self.manwhitney_p_value}")

    @classmethod
    def from_config(cls, config: dict) -> "PriorValidationConfig":
        """Extract prior validation configuration."""
        return cls(
            validate=config["validate_prior"],
            method=config["prior_validation_method"],
            n_samples=(int(config["n_prior_validation_samples"]) if config["n_prior_validation_samples"] is not None else None),
            manwhitney_p_value=(float(config["prior_validation_manwhitney_p"]) if config["prior_validation_manwhitney_p"] is not None else None),
            difference_threshold=(float(config["prior_validation_difference_threshold"]) if config["prior_validation_difference_threshold"] is not None else None),
        )


def extract_optimization_approach(config: dict) -> tuple[bool, bool]:
    """Extract and validate optimization approach."""
    dynabo = config["dynabo"]
    pibo = config["pibo"]
    assert dynabo ^ pibo, "Either DynaBO or PiBO must be True"
    return dynabo, pibo
