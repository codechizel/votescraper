"""Model specification dataclasses for hierarchical IRT experiments.

Factors the bill discrimination (beta) prior into a frozen dataclass that both
production and experiments consume. Production uses PRODUCTION_BETA as the default;
experiments pass alternative specs to the same model-building functions.

See docs/experiment-framework-deep-dive.md for design rationale.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BetaPriorSpec:
    """Specification for the bill discrimination (beta) prior.

    The distribution field selects the PyMC distribution family.
    The params dict is unpacked as keyword arguments to the distribution constructor.

    Examples:
        BetaPriorSpec("normal", {"mu": 0, "sigma": 1})            # production default
        BetaPriorSpec("lognormal", {"mu": 0, "sigma": 0.5})        # soft positive (PyMC transform)
        BetaPriorSpec("lognormal_reparam", {"mu": 0, "sigma": 1})  # exp reparameterized
        BetaPriorSpec("halfnormal", {"sigma": 1})                  # hard zero floor
    """

    distribution: str
    params: dict[str, float]

    def build(self, n_votes: int, dims: str = "vote"):
        """Instantiate the PyMC distribution inside an active model context.

        Must be called inside a ``with pm.Model():`` block.

        Args:
            n_votes: Number of bills/votes (shape parameter).
            dims: PyMC dimension name for the variable.

        Returns:
            PyMC distribution variable named "beta".

        Raises:
            ValueError: If distribution is not one of the supported types.
        """
        import pymc as pm
        import pytensor.tensor as pt

        match self.distribution:
            case "normal":
                return pm.Normal("beta", shape=n_votes, dims=dims, **self.params)
            case "lognormal":
                return pm.LogNormal("beta", shape=n_votes, dims=dims, **self.params)
            case "lognormal_reparam":
                # Reparameterized LogNormal: sample log_beta ~ Normal(mu, sigma)
                # in unconstrained space, then beta = exp(log_beta). Mathematically
                # identical to LogNormal but avoids PyMC's internal log transform,
                # which creates catastrophic curvature near beta=0.
                # See docs/joint-model-deep-dive.md for the diagnosis.
                log_beta = pm.Normal("log_beta", shape=n_votes, dims=dims, **self.params)
                return pm.Deterministic("beta", pt.exp(log_beta), dims=dims)
            case "halfnormal":
                return pm.HalfNormal("beta", shape=n_votes, dims=dims, **self.params)
            case _:
                msg = (
                    f"Unknown beta prior distribution: {self.distribution!r}. "
                    f"Supported: normal, lognormal, lognormal_reparam, halfnormal"
                )
                raise ValueError(msg)

    def describe(self) -> str:
        """Human-readable description for logs and reports.

        Returns:
            String like "Normal(mu=0, sigma=1)" or "LogNormal_reparam(mu=0, sigma=1)".
        """
        name = self.distribution.capitalize()
        if self.distribution == "lognormal":
            name = "LogNormal"
        elif self.distribution == "lognormal_reparam":
            name = "LogNormal_reparam"
        elif self.distribution == "halfnormal":
            name = "HalfNormal"
        param_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{name}({param_str})"


# Production default: matches the hardcoded pm.Normal("beta", mu=0, sigma=1, ...)
# that has been in build_per_chamber_model() and build_joint_model() since ADR-0017.
PRODUCTION_BETA = BetaPriorSpec("normal", {"mu": 0, "sigma": 1})

# Joint model beta: reparameterized LogNormal eliminates reflection mode multimodality
# without the boundary geometry that caused 2,041 divergences with pm.LogNormal.
# exp(Normal(0, 1)) has prior median=1.0, 95% interval=[0.14, 7.39] — wide enough
# for bills with near-zero discrimination. See docs/joint-model-deep-dive.md.
JOINT_BETA = BetaPriorSpec("lognormal_reparam", {"mu": 0, "sigma": 1})
