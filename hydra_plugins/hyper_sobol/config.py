"""Config for HyperSobol sweeper."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

from hydra.core.config_store import ConfigStore


@dataclass
class HyperSobolConfig:
    """Config for HyperSobol sweeper."""

    _target_: str = "hydra_plugins.hypersweeper.hypersweeper.Hypersweeper"
    opt_constructor: str = "hydra_plugins.hyper_sobol.hyper_sobol.make_sobol"
    search_space: Optional[Dict] = field(default_factory=dict)
    resume: Union[str, bool] = False
    budget: Optional[Any] = None
    n_trials: Optional[int] = None
    budget_variable: Optional[str] = None
    loading_variable: Optional[str] = None
    saving_variable: Optional[str] = None
    sweeper_kwargs: Optional[Dict] = field(default_factory=dict)


ConfigStore.instance().store(
    group="hydra/sweeper",
    name="HyperSobol",
    node=HyperSobolConfig,
    provider="hypersweeper",
)
