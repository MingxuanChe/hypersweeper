"""HyperSobol: Sobol Sequence Sampling in Hypersweeper."""

from __future__ import annotations

import numpy as np
from ConfigSpace import Configuration
from hydra_plugins.hypersweeper import Info
from scipy.stats.qmc import Sobol


class HyperSobol:
    """Sobol Sequence Sampling.
    
    Uses quasi-random Sobol sequences for low-discrepancy sampling of the configuration space.
    Supports partitioning the sequence into subsequences for parallel execution.
    """

    def __init__(
        self,
        configspace,
        seed: int = 42,
        n_sub: int = 1,
        id_sub: int = 0,
        max_samples: int = 10000,
    ) -> None:
        """Initialize the Sobol sampler.
        
        Parameters
        ----------
        configspace : ConfigurationSpace
            The configuration space to sample from.
        seed : int, optional
            Random seed for the Sobol sequence, by default 42.
        n_sub : int, optional
            Number of partitions to divide the Sobol sequence into, by default 1 (no partitioning).
        id_sub : int, optional
            Which partition to use (0-indexed), by default 0.
            Must be in range [0, n_sub).
        max_samples : int, optional
            Maximum number of samples to pre-generate for this partition, by default 10000.
        """
        self.configspace = configspace
        self.seed = seed
        self.n_sub = n_sub
        self.id_sub = id_sub
        self.max_samples = max_samples
        
        if id_sub < 0 or id_sub >= n_sub:
            raise ValueError(f"id_sub must be in range [0, {n_sub}), got {id_sub}")
        
        # Get dimensionality of the configuration space
        self.n_dims = len(list(configspace.values()))
        
        # Pre-generate Sobol samples for this partition
        self._pregenerate_samples()
        
        # Counter for samples within this partition
        self.sample_count = 0
        
    def _pregenerate_samples(self):
        """Pre-generate Sobol samples for this partition."""
        # Initialize Sobol sequence generator
        sobol_engine = Sobol(d=self.n_dims, scramble=True, seed=self.seed)
        
        # Total samples needed: enough to cover max_samples for this partition
        total_needed = self.id_sub + self.max_samples * self.n_sub
        
        # Generate all samples
        all_samples = sobol_engine.random(total_needed)
        
        # Extract samples for this partition (every n_sub-th sample starting from id_sub)
        self.partition_samples = all_samples[self.id_sub::self.n_sub]
        
    def ask(self):
        """Sample next configuration from the partitioned Sobol sequence.
        
        Returns
        -------
        tuple
            (Info object with configuration, False, False)
        """
        if self.sample_count >= len(self.partition_samples):
            raise ValueError(
                f"Requested more samples ({self.sample_count + 1}) than pre-generated "
                f"({len(self.partition_samples)}). Increase max_samples parameter."
            )
        
        # Get the next sample for this partition
        sobol_sample = self.partition_samples[self.sample_count]
        
        # Convert Sobol sample (in [0, 1]^d) to a configuration
        config = self._sobol_to_config(sobol_sample)
        
        self.sample_count += 1
        
        info = Info(config=config, budget=None, load_path=None, seed=None)
        return info, False, False

    def _sobol_to_config(self, sobol_sample: np.ndarray) -> Configuration:
        """Convert a Sobol sample to a configuration.
        
        Parameters
        ----------
        sobol_sample : np.ndarray
            Sobol sample in [0, 1]^d.
            
        Returns
        -------
        Configuration
            Configuration object.
        """
        # Create a configuration from the Sobol sample
        # ConfigSpace expects values in the appropriate ranges for each hyperparameter
        config_dict = {}
        
        for i, hp in enumerate(list(self.configspace.values())):
            value = sobol_sample[i]
            
            # Transform [0, 1] to the appropriate range based on hyperparameter type
            from ConfigSpace.hyperparameters import (
                CategoricalHyperparameter,
                Constant,
                NumericalHyperparameter,
                OrdinalHyperparameter,
            )
            
            if isinstance(hp, CategoricalHyperparameter):
                # Map to categorical choices
                idx = int(value * len(hp.choices))
                idx = min(idx, len(hp.choices) - 1)  # Ensure we don't exceed bounds
                config_dict[hp.name] = hp.choices[idx]
            elif isinstance(hp, OrdinalHyperparameter):
                # Map to ordinal choices
                idx = int(value * len(hp.sequence))
                idx = min(idx, len(hp.sequence) - 1)
                config_dict[hp.name] = hp.sequence[idx]
            elif isinstance(hp, NumericalHyperparameter):
                # Map to numerical range, respecting log scale if applicable
                if hp.log:
                    # Log scale
                    log_lower = np.log(hp.lower)
                    log_upper = np.log(hp.upper)
                    log_value = log_lower + value * (log_upper - log_lower)
                    numeric_value = np.exp(log_value)
                else:
                    # Linear scale
                    numeric_value = hp.lower + value * (hp.upper - hp.lower)
                
                # Handle integer hyperparameters
                from ConfigSpace.hyperparameters import IntegerHyperparameter
                if isinstance(hp, IntegerHyperparameter):
                    numeric_value = int(np.round(numeric_value))
                
                config_dict[hp.name] = numeric_value
            elif isinstance(hp, Constant):
                config_dict[hp.name] = hp.value
            else:
                raise ValueError(f"Unsupported hyperparameter type: {type(hp)}")
        
        # Use allow_inactive_with_values=True to handle conditional hyperparameters
        # This allows values for inactive hyperparameters instead of raising InactiveHyperparameterSetError
        # ConfigSpace will handle inactive hyperparameters appropriately during configuration usage
        return Configuration(self.configspace, values=config_dict, allow_inactive_with_values=True)

    def tell(self, info, value):
        """Do nothing for Sobol sampling (non-adaptive)."""
        pass

    def finish_run(self, output_path):
        """Save partition information for later merging."""
        import json
        from pathlib import Path
        
        output_path = Path(output_path)
        partition_info = {
            "seed": self.seed,
            "n_sub": self.n_sub,
            "id_sub": self.id_sub,
            "n_samples": self.sample_count,
        }
        
        partition_file = output_path / f"sobol_partition_{self.id_sub}_of_{self.n_sub}.json"
        with open(partition_file, "w") as f:
            json.dump(partition_info, f, indent=2)


def make_sobol(configspace, hyper_sobol_args):
    """Make a Sobol sampler instance for optimization.
    
    Parameters
    ----------
    configspace : ConfigurationSpace
        The configuration space to sample from.
    hyper_sobol_args : dict
        Arguments for the Sobol sampler:
        - seed: int, random seed (default: 42)
        - n_sub: int, number of partitions (default: 1)
        - id_sub: int, partition index (default: 0)
        - max_samples: int, maximum samples to pre-generate (default: 10000)
        
    Returns
    -------
    HyperSobol
        Sobol sampler instance.
    """
    # Set defaults
    defaults = {"seed": 42, "n_sub": 1, "id_sub": 0, "max_samples": 10000}
    defaults.update(hyper_sobol_args)
    return HyperSobol(configspace, **defaults)
