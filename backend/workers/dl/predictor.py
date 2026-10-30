"""DL tier prediction: one molecule and one target in, one labelled result payload out (MSP-26).

Owner: Nishitha. Requirements: FR-015..FR-020, NFR-002, NFR-007, NFR-008, NFR-011.

predict_smiles() turns the stored canonical SMILES into a graph with the same function the
models were trained with (backend.workers.dl.graph), then returns:

  payload       {"tier": "dl", "target": {...}, "activity": {...}, "baseline": {...}}
                activity = ensemble mean and SD with the target's own unit; baseline = one value
  model_version "<card name>@<version>", for example dl_egfr_gnn@1.0.0
  params_hash   SHA-256 of the card identity, every model file hash, both feature signatures,
                the target and the SD setting

One predictor per target is loaded once per worker process and verified again if a file changes
on disk. Models live in <MODELS_DIR>/dl/ (the folder train_gnn writes to), cards in
backend/registry/cards/. A problem raises ModelLoadError, which the task turns into
MODEL_UNAVAILABLE; there is no fallback model.
"""
# RDKit is a C extension, so pylint cannot see its members.
# pylint: disable=no-member,c-extension-no-member
import os
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional

from rdkit import Chem

from backend.core import labels
from backend.core.config import load_targets
from backend.registry.loader import ModelLoadError
from backend.workers.dl import baseline, ensemble, graph
from backend.workers.ml import featuriser, model_cards

DL_TIER = "dl"
TASK = "target_activity"
MODELS_SUBFOLDER = "dl"
DEFAULT_MODEL_VERSION = "1.0.0"
MODEL_VERSION_ENV = "DL_MODEL_VERSION"
MAX_MODEL_VERSION_LEN = 50  # results.model_version column size


def model_version() -> str:
    """DL model version to serve (the cards on main are 1.0.0; override with DL_MODEL_VERSION)."""
    return os.environ.get(MODEL_VERSION_ENV) or DEFAULT_MODEL_VERSION


def slug_of(target_id: str) -> str:
    """File-name form of a target id. Must match training.train_gnn.slug_of (a test checks it)."""
    return "".join(ch if ch.isalnum() else "_" for ch in target_id.lower())


def card_name(target_id: str) -> str:
    """Model name of a target's card, for example dl_egfr_gnn."""
    return f"dl_{slug_of(target_id)}_gnn"


def find_target(target_id: Optional[str], targets: Optional[List[Mapping]] = None) -> Mapping:
    """The configured target with this id (ModelLoadError if there is none)."""
    if not target_id:
        raise ModelLoadError("The job has no target.")
    configured = targets if targets is not None else (load_targets() or {}).get("targets") or []
    for target in configured:
        if target.get("id") == target_id:
            for key in ("id", "name", "chembl_id", "unit"):
                if not target.get(key):
                    raise ModelLoadError("The target configuration is incomplete.")
            return target
    raise ModelLoadError("The target is not configured.")


@dataclass(frozen=True)
class DlResult:
    """What the task stores: the payload plus the two values saved beside it."""

    payload: Dict
    model_version: str
    params_hash: str


def check_card(card: Mapping, target: Mapping) -> None:
    """Refuse a card that does not belong to this target, unit or graph definition."""
    info = card.get("graph")
    if not isinstance(info, dict) or info.get("signature") != graph.graph_signature():
        raise ModelLoadError("Model was built with a different graph definition than this code uses.")
    if info.get("n_atom_features") != graph.N_ATOM_FEATURES or info.get("n_bond_features") != graph.N_BOND_FEATURES:
        raise ModelLoadError("Model card has different graph sizes than this code uses.")
    recorded = card.get("target")
    if not isinstance(recorded, dict) or recorded.get("id") != target["id"] \
            or recorded.get("chembl_id") != target["chembl_id"]:
        raise ModelLoadError("Model card belongs to a different target.")
    if card.get("unit") != target["unit"]:
        raise ModelLoadError("Model card unit differs from the target configuration.")


class DlPredictor:
    """The loaded ensemble and baseline for one target."""

    def __init__(self, target: Mapping, card: dict, members: ensemble.TargetEnsemble,
                 base: baseline.BaselineModel, files: model_cards.VerifiedFiles):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self.target = {key: target[key] for key in ("id", "name", "chembl_id", "unit")}
        self.card = card
        self.ensemble = members
        self.baseline = base
        self.files = files
        self.model_version = f"{card['name']}@{card['version']}"
        if len(self.model_version) > MAX_MODEL_VERSION_LEN:
            raise ModelLoadError("Model version string is too long to store.")
        self.params_hash = model_cards.stable_hash({
            "card": model_cards.card_identity(card),
            "graph": graph.graph_signature(),
            "featuriser": featuriser.feature_signature(),
            "target": {key: self.target[key] for key in ("id", "chembl_id", "unit")},
            "sd_ddof": members.ddof,
        })

    @classmethod
    def load(cls, target: Mapping, models_dir=None, version: str = "", cards_dir=None) -> "DlPredictor":
        """Verify and load the ensemble and baseline of one target."""
        card = model_cards.read_card(
            model_cards.card_path(card_name(target["id"]), version or model_version(), cards_dir),
            TASK, tier=DL_TIER)
        check_card(card, target)
        member_names = ensemble.member_files(card)
        base_name = baseline.baseline_file(card)
        ddof = ensemble.sd_ddof(card)
        folder = model_cards.models_folder(models_dir, MODELS_SUBFOLDER)
        files = model_cards.verify_files(card, [*member_names, base_name], folder)
        return cls(
            target, card,
            ensemble.TargetEnsemble.load([files.paths[name] for name in member_names], ddof),
            baseline.BaselineModel.load(files.paths[base_name], card),
            files,
        )

    def still_current(self) -> bool:
        """False if any model file changed on disk since it was verified."""
        return self.files.unchanged()

    def predict_smiles(self, smiles: str) -> DlResult:
        """Predict for one standardised SMILES (ValueError if it cannot be parsed or is unsupported)."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("The stored structure could not be parsed.")
        mean, sd, values = self.ensemble.predict(graph.mol_to_graph(mol))
        reference = self.baseline.predict(featuriser.featurise_mol(mol))
        unit = self.target["unit"]
        payload = {
            "tier": DL_TIER,
            "target": {key: self.target[key] for key in ("id", "name", "chembl_id")},
            "activity": {
                "mean": mean,
                "sd": sd,
                "n_models": len(values),
                "member_values": values,
                "model_name": self.card["name"],
                **labels.activity_labels(unit, self.model_version),
            },
            "baseline": {
                "value": reference,
                "algorithm": "gradient-boosted trees (XGBoost)",
                "model_name": self.card["name"],
                **labels.baseline_labels(unit, self.model_version),
            },
        }
        return DlResult(payload, self.model_version, self.params_hash)


_CACHE: Dict[str, DlPredictor] = {}


def get_predictor(target_id: Optional[str], models_dir=None, cards_dir=None,
                  targets: Optional[List[Mapping]] = None) -> DlPredictor:
    """The shared predictor of one target in this worker process.

    Loaded on first use. If a model file changed since it was loaded, everything is verified and
    loaded again; a failed load is not cached, so fixing the file is enough.
    """
    target = find_target(target_id, targets)
    cached = _CACHE.get(target["id"])
    if cached is not None and not cached.still_current():
        _CACHE.pop(target["id"], None)
        cached = None
    if cached is None:
        cached = DlPredictor.load(target, models_dir, cards_dir=cards_dir)
        _CACHE[target["id"]] = cached
    return cached


def reset_predictor() -> None:
    """Forget every cached predictor (tests, and after replacing the model files)."""
    _CACHE.clear()
