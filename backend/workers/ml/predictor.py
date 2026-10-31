"""ML tier prediction: one molecule in, one labelled result payload out (MSP-23).

Owner: Nishitha. Requirements: FR-009, FR-010, FR-011, FR-012, BR-006, NFR-007, NFR-008.

predict_smiles() featurises the stored canonical SMILES with the same featuriser the
models were trained with, then returns:

  payload       {"tier": "ml", "solubility": {...}, "tox21": {...}}  (JSON-safe, labelled)
  model_version "<card name>@<version>" for both model families, joined with "+"
  params_hash   SHA-256 of every card value and file hash that affects the output

Models are loaded once per worker process and re-verified if a file changes on disk.
MSP-24 adds three blocks to the payload (existing fields are unchanged):

  "alerts"   PAINS and Brenk matches            (backend/workers/ml/alerts.py)
  "rules"    Lipinski and Veber violations      (backend/workers/ml/alerts.py)
  "domain"   applicability domain per model family and overall (backend/workers/ml/domain.py)

The domain reference files are checked like model files. If they cannot be loaded the models still
run, and the domain block says "available": false and in_domain null: never an invented 'inside'.
Pure RDKit, numpy and XGBoost: no Flask, database or Celery imports.
"""
# RDKit is a C extension, so pylint cannot see its members.
# pylint: disable=no-member,c-extension-no-member
import logging
from dataclasses import dataclass
from typing import Dict, Optional

from rdkit import Chem

from backend.core import labels
from backend.core.config import Settings
from backend.registry.loader import ModelLoadError
from backend.workers.ml import alerts, domain, featuriser, model_cards, solubility, tox21

log = logging.getLogger("msp.worker.ml")

MAX_MODEL_VERSION_LEN = 50  # results.model_version column size
FAMILIES = ("solubility", "tox21")


@dataclass(frozen=True)
class MlResult:
    """What the task stores: the payload plus the two values saved beside it."""

    payload: Dict
    model_version: str
    params_hash: str


class MlPredictor:
    """The loaded solubility and Tox21 models."""

    def __init__(self, sol: solubility.SolubilityModel, tox: tox21.Tox21Models,
                 references: Optional[Dict[str, domain.DomainReference]] = None,
                 ad_threshold: Optional[float] = None):
        self.solubility = sol
        self.tox21 = tox
        self.references = references or {}
        self.ad_threshold = float(Settings.from_env().ad_tanimoto_min if ad_threshold is None else ad_threshold)
        self.model_version = (
            f"{sol.card['name']}@{sol.version}+{tox.card['name']}@{tox.version}"
        )
        if len(self.model_version) > MAX_MODEL_VERSION_LEN:
            raise ModelLoadError("Model version string is too long to store.")
        self.params_hash = model_cards.stable_hash({
            "featuriser": featuriser.feature_signature(),
            "solubility": sol.hash_inputs(),
            "tox21": tox.hash_inputs(),
            "alerts": alerts.identity(),
            "domain": {"threshold": self.ad_threshold,
                       "references": {name: self.references[name].identity if name in self.references else None
                                      for name in FAMILIES}},
        })

    @classmethod
    def load(cls, models_dir=None, version: str = "", cards_dir=None) -> "MlPredictor":
        """Verify and load both model families (raises ModelLoadError, never falls back).

        The domain reference files are loaded too, but a problem with them only turns the domain
        check off (reported in the payload); it does not stop the models.
        """
        folder = model_cards.models_folder(models_dir)
        sol = solubility.SolubilityModel.load(
            model_cards.card_path(solubility.CARD_NAME, version, cards_dir), folder)
        tox = tox21.Tox21Models.load(model_cards.card_path(tox21.CARD_NAME, version, cards_dir), folder)
        references = {}
        for name, model in (("solubility", sol), ("tox21", tox)):
            try:
                references[name] = domain.DomainReference.load(model.card, folder)
            except (ModelLoadError, OSError) as exc:
                log.error("ml domain reference data for %s unavailable: %s", name, type(exc).__name__)
        return cls(sol, tox, references)

    def still_current(self) -> bool:
        """False if any model file changed on disk since it was verified."""
        return (self.solubility.files.unchanged() and self.tox21.files.unchanged()
                and all(reference.still_current() for reference in self.references.values()))

    def domain_block(self, features) -> Dict:
        """Applicability domain per model family, the overall flag and the warning text."""
        families = {}
        for name in FAMILIES:
            reference = self.references.get(name)
            families[name] = (reference.check(features, self.ad_threshold) if reference
                              else domain.unavailable(self.ad_threshold))
        verdict = domain.overall(families)
        warning = {False: labels.WARNING_OUT_OF_DOMAIN, None: labels.WARNING_DOMAIN_UNAVAILABLE}.get(verdict)
        return {
            "in_domain": verdict,
            "warning": warning,
            "families": families,
            **labels.domain_labels(f"ml_domain@{self.solubility.version}"),
        }

    def predict_smiles(self, smiles: str) -> MlResult:
        """Predict for one standardised SMILES (ValueError if it cannot be parsed)."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("The stored structure could not be parsed.")
        features = featuriser.featurise_mol(mol)
        payload = {
            "tier": model_cards.ML_TIER,
            "solubility": self.solubility.predict(features),
            "tox21": self.tox21.predict(features),
            "alerts": alerts.structural_alerts(mol),
            "rules": alerts.property_rules(mol),
            "domain": self.domain_block(features),
        }
        return MlResult(payload, self.model_version, self.params_hash)


_CACHE: Optional[MlPredictor] = None


def get_predictor(models_dir=None, cards_dir=None) -> MlPredictor:
    """The shared predictor for this worker process.

    Loaded on first use. If a model file changed since it was loaded, everything is
    verified and loaded again; a failed load is not cached, so fixing the file is enough.
    """
    global _CACHE  # pylint: disable=global-statement
    if _CACHE is not None and not _CACHE.still_current():
        _CACHE = None
    if _CACHE is None:
        _CACHE = MlPredictor.load(models_dir, cards_dir=cards_dir)
    return _CACHE


def reset_predictor() -> None:
    """Forget the cached predictor (tests, and after replacing the model files)."""
    global _CACHE  # pylint: disable=global-statement
    _CACHE = None
