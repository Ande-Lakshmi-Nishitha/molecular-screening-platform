"""ML tier prediction: one molecule in, one labelled result payload out (MSP-23).

Owner: Nishitha. Requirements: FR-009, FR-010, FR-011, FR-012, BR-006, NFR-007, NFR-008.

predict_smiles() featurises the stored canonical SMILES with the same featuriser the
models were trained with, then returns:

  payload       {"tier": "ml", "solubility": {...}, "tox21": {...}}  (JSON-safe, labelled)
  model_version "<card name>@<version>" for both model families, joined with "+"
  params_hash   SHA-256 of every card value and file hash that affects the output

Models are loaded once per worker process and re-verified if a file changes on disk.
Alerts, Lipinski/Veber counts and the domain flag join the payload in MSP-24.
Pure RDKit, numpy and XGBoost: no Flask, database or Celery imports.
"""
# RDKit is a C extension, so pylint cannot see its members.
# pylint: disable=no-member,c-extension-no-member
from dataclasses import dataclass
from typing import Dict, Optional

from rdkit import Chem

from backend.registry.loader import ModelLoadError
from backend.workers.ml import featuriser, model_cards, solubility, tox21

MAX_MODEL_VERSION_LEN = 50  # results.model_version column size


@dataclass(frozen=True)
class MlResult:
    """What the task stores: the payload plus the two values saved beside it."""

    payload: Dict
    model_version: str
    params_hash: str


class MlPredictor:
    """The loaded solubility and Tox21 models."""

    def __init__(self, sol: solubility.SolubilityModel, tox: tox21.Tox21Models):
        self.solubility = sol
        self.tox21 = tox
        self.model_version = (
            f"{sol.card['name']}@{sol.version}+{tox.card['name']}@{tox.version}"
        )
        if len(self.model_version) > MAX_MODEL_VERSION_LEN:
            raise ModelLoadError("Model version string is too long to store.")
        self.params_hash = model_cards.stable_hash({
            "featuriser": featuriser.feature_signature(),
            "solubility": sol.hash_inputs(),
            "tox21": tox.hash_inputs(),
        })

    @classmethod
    def load(cls, models_dir=None, version: str = "", cards_dir=None) -> "MlPredictor":
        """Verify and load both model families (raises ModelLoadError, never falls back)."""
        folder = model_cards.models_folder(models_dir)
        return cls(
            solubility.SolubilityModel.load(
                model_cards.card_path(solubility.CARD_NAME, version, cards_dir), folder),
            tox21.Tox21Models.load(model_cards.card_path(tox21.CARD_NAME, version, cards_dir), folder),
        )

    def still_current(self) -> bool:
        """False if any model file changed on disk since it was verified."""
        return self.solubility.files.unchanged() and self.tox21.files.unchanged()

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
