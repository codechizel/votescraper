"""Analysis pipeline for Tallgrass.

Single-biennium pipeline (01-25):
  01_eda           — Exploratory Data Analysis
  02_pca           — Principal Component Analysis
  03_mca           — Multiple Correspondence Analysis
  04_umap          — UMAP Visualization
  05_irt           — Bayesian IRT (1D)
  06_irt_2d        — 2D Bayesian IRT
  07_hierarchical  — Hierarchical Bayesian IRT
  08_ppc           — Posterior Predictive Checks + LOO-CV
  09_clustering    — Voting Bloc Detection
  10_lca           — Latent Class Analysis
  11_network       — Legislator Network
  12_bipartite     — Bipartite Network
  13_indices       — Legislative Indices
  14_beta_binomial — Bayesian Party Loyalty
  15_prediction    — Vote Prediction
  16_wnominate     — W-NOMINATE + OC Validation
  17_external_validation — Shor-McCarty Validation
  18_dime          — DIME/CFscore Validation
  19_tsa           — Time Series Analysis
  20_bill_text     — Bill Text NLP
  21_tbip          — Text-Based Ideal Points
  22_issue_irt     — Issue-Specific Ideal Points
  23_model_legislation — Model Legislation Detection
  24_synthesis     — Narrative Synthesis
  25_profiles      — Legislator Profiles

Cross-biennium pipeline (26-27):
  26_cross_session — Cross-Biennium Validation
  27_dynamic_irt   — Dynamic Ideal Points

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
    "mca": "03_mca",
    "mca_report": "03_mca",
    "umap_viz": "04_umap",
    "umap_report": "04_umap",
    "irt": "05_irt",
    "irt_report": "05_irt",
    "irt_2d": "06_irt_2d",
    "irt_2d_report": "06_irt_2d",
    "hierarchical": "07_hierarchical",
    "hierarchical_report": "07_hierarchical",
    "model_spec": "07_hierarchical",
    "irt_linking": "07_hierarchical",
    "hierarchical_2d": "07b_hierarchical_2d",
    "hierarchical_2d_report": "07b_hierarchical_2d",
    "ppc": "08_ppc",
    "ppc_data": "08_ppc",
    "ppc_report": "08_ppc",
    "clustering": "09_clustering",
    "clustering_report": "09_clustering",
    "lca": "10_lca",
    "lca_report": "10_lca",
    "network": "11_network",
    "network_report": "11_network",
    "bipartite": "12_bipartite",
    "bipartite_report": "12_bipartite",
    "indices": "13_indices",
    "indices_report": "13_indices",
    "beta_binomial": "14_beta_binomial",
    "beta_binomial_report": "14_beta_binomial",
    "prediction": "15_prediction",
    "prediction_report": "15_prediction",
    "nlp_features": "15_prediction",
    "wnominate": "16_wnominate",
    "wnominate_data": "16_wnominate",
    "wnominate_report": "16_wnominate",
    "external_validation": "17_external_validation",
    "external_validation_report": "17_external_validation",
    "external_validation_data": "17_external_validation",
    "external_validation_dime": "18_dime",
    "external_validation_dime_report": "18_dime",
    "external_validation_dime_data": "18_dime",
    "tsa": "19_tsa",
    "tsa_report": "19_tsa",
    "tsa_r_data": "19_tsa",
    "bill_text": "20_bill_text",
    "bill_text_data": "20_bill_text",
    "bill_text_classify": "20_bill_text",
    "bill_text_report": "20_bill_text",
    "tbip": "21_tbip",
    "tbip_data": "21_tbip",
    "tbip_report": "21_tbip",
    "issue_irt": "22_issue_irt",
    "issue_irt_data": "22_issue_irt",
    "issue_irt_report": "22_issue_irt",
    "model_legislation": "23_model_legislation",
    "model_legislation_data": "23_model_legislation",
    "model_legislation_report": "23_model_legislation",
    "synthesis": "24_synthesis",
    "synthesis_data": "24_synthesis",
    "synthesis_report": "24_synthesis",
    "synthesis_detect": "24_synthesis",
    "profiles": "25_profiles",
    "profiles_report": "25_profiles",
    "profiles_data": "25_profiles",
    "cross_session": "26_cross_session",
    "cross_session_report": "26_cross_session",
    "cross_session_data": "26_cross_session",
    "dynamic_irt": "27_dynamic_irt",
    "dynamic_irt_data": "27_dynamic_irt",
    "dynamic_irt_report": "27_dynamic_irt",
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
