"""Analysis pipeline for Tallgrass.

Pipeline phases (in order):
  01_eda           — Exploratory Data Analysis
  02_pca           — Principal Component Analysis
  02c_mca          — Multiple Correspondence Analysis
  03_umap          — UMAP dimensionality reduction
  04_irt           — Bayesian IRT ideal points
  04b_irt_2d       — 2D Bayesian IRT ideal points (experimental)
  04c_ppc          — Posterior predictive checks + LOO-CV model comparison
  05_clustering    — Voting bloc detection
  05b_lca          — Latent Class Analysis (Bernoulli mixture)
  06_network       — Legislator network analysis
  06b_network_bipartite — Bipartite bill-legislator network
  07_indices       — Classical political science indices
  08_prediction    — Vote prediction (XGBoost)
  09_beta_binomial — Bayesian party loyalty
  10_hierarchical  — Hierarchical IRT
  11_synthesis     — Narrative synthesis
  12_profiles      — Legislator deep dives
  13_cross_session — Cross-biennium validation
  14_external_validation — Shor-McCarty comparison
  14b_external_validation_dime — DIME/CFscore comparison
  15_tsa              — Time series analysis (drift + changepoints)
  16_dynamic_irt      — Dynamic ideal points (state-space IRT)
  17_wnominate        — W-NOMINATE + Optimal Classification validation
  18_bill_text         — Bill text analysis (BERTopic + CAP classification)
  18b_tbip             — Text-based ideal points (embedding-vote approach)

Shared infrastructure at root: run_context.py, report.py

Uses a PEP 302 meta-path finder so that ``from analysis.eda import X``
transparently loads ``analysis.01_eda.eda``.  Zero import changes needed.
"""

import importlib
import sys
import types
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec

_MODULE_MAP: dict[str, str] = {
    "eda": "01_eda",
    "eda_report": "01_eda",
    "bill_lifecycle": "01_eda",
    "geographic": "01_eda",
    "pca": "02_pca",
    "pca_report": "02_pca",
    "mca": "02c_mca",
    "mca_report": "02c_mca",
    "umap_viz": "03_umap",
    "umap_report": "03_umap",
    "irt": "04_irt",
    "irt_report": "04_irt",
    "irt_2d": "04b_irt_2d",
    "irt_2d_report": "04b_irt_2d",
    "clustering": "05_clustering",
    "clustering_report": "05_clustering",
    "lca": "05b_lca",
    "lca_report": "05b_lca",
    "network": "06_network",
    "network_report": "06_network",
    "bipartite": "06b_network_bipartite",
    "bipartite_report": "06b_network_bipartite",
    "indices": "07_indices",
    "indices_report": "07_indices",
    "prediction": "08_prediction",
    "prediction_report": "08_prediction",
    "nlp_features": "08_prediction",
    "beta_binomial": "09_beta_binomial",
    "beta_binomial_report": "09_beta_binomial",
    "hierarchical": "10_hierarchical",
    "hierarchical_report": "10_hierarchical",
    "model_spec": "10_hierarchical",
    "irt_linking": "10_hierarchical",
    "synthesis": "11_synthesis",
    "synthesis_data": "11_synthesis",
    "synthesis_report": "11_synthesis",
    "synthesis_detect": "11_synthesis",
    "profiles": "12_profiles",
    "profiles_report": "12_profiles",
    "profiles_data": "12_profiles",
    "cross_session": "13_cross_session",
    "cross_session_report": "13_cross_session",
    "cross_session_data": "13_cross_session",
    "external_validation": "14_external_validation",
    "external_validation_report": "14_external_validation",
    "external_validation_data": "14_external_validation",
    "external_validation_dime": "14b_external_validation_dime",
    "external_validation_dime_report": "14b_external_validation_dime",
    "external_validation_dime_data": "14b_external_validation_dime",
    "tsa": "15_tsa",
    "tsa_report": "15_tsa",
    "tsa_r_data": "15_tsa",
    "dynamic_irt": "16_dynamic_irt",
    "dynamic_irt_data": "16_dynamic_irt",
    "dynamic_irt_report": "16_dynamic_irt",
    "ppc": "04c_ppc",
    "ppc_data": "04c_ppc",
    "ppc_report": "04c_ppc",
    "wnominate": "17_wnominate",
    "wnominate_data": "17_wnominate",
    "wnominate_report": "17_wnominate",
    "irt_beta_experiment": "experimental",
    "bill_text": "18_bill_text",
    "bill_text_data": "18_bill_text",
    "bill_text_classify": "18_bill_text",
    "bill_text_report": "18_bill_text",
    "tbip": "18b_tbip",
    "tbip_data": "18b_tbip",
    "tbip_report": "18b_tbip",
}


class _AliasLoader:
    """Loader that imports the real module and registers it under the alias."""

    def __init__(self, real_name: str) -> None:
        self.real_name = real_name

    def create_module(self, spec: ModuleSpec) -> types.ModuleType | None:
        return None  # use default semantics

    def exec_module(self, module: types.ModuleType) -> None:
        real = importlib.import_module(self.real_name)
        # Copy all attributes from the real module into the alias
        module.__dict__.update(real.__dict__)
        module.__file__ = real.__file__
        module.__loader__ = real.__loader__
        if hasattr(real, "__path__"):
            module.__path__ = real.__path__


class _AnalysisRedirectFinder(MetaPathFinder):
    """Redirect ``analysis.<name>`` imports to ``analysis.<NN_subdir>.<name>``."""

    def find_spec(
        self,
        fullname: str,
        path: object = None,
        target: types.ModuleType | None = None,
    ) -> ModuleSpec | None:
        parts = fullname.split(".")
        if len(parts) == 2 and parts[0] == "analysis" and parts[1] in _MODULE_MAP:
            name = parts[1]
            subpkg = _MODULE_MAP[name]
            real = f"analysis.{subpkg}.{name}"
            return ModuleSpec(fullname, _AliasLoader(real))
        return None


sys.meta_path.insert(0, _AnalysisRedirectFinder())
