"""Five TorchScript models, mean and SD.

Owner: Nishitha. Requirements: FR-017, FR-018, FR-020.

One ensemble per target. Each member is a TorchScript file written by training/train_gnn.py;
it takes the arrays of one molecule graph and answers in the target's own unit (pIC50 or pKi),
because the training mean and SD are stored inside the file. The reported value is the mean
of the members and the reported spread is their sample SD (ddof from the card, 1).

Files are verified (no pickle, SHA-256 from the card) by backend.workers.ml.model_cards before
torch ever opens them; TorchScript files carry their own code and need no training source.
"""
import math
from typing import List, Mapping, Sequence, Tuple

import numpy as np
import torch  # pylint: disable=import-error

from backend.registry.loader import ModelLoadError
from backend.workers.dl import graph

MEMBER_ROLE = "gnn_member"
MIN_MEMBERS = 2          # an SD needs at least two values; production cards have five


def member_files(card: Mapping) -> List[str]:
    """File names of the ensemble members, in member order (1, 2, ...)."""
    entries = [entry for entry in card["artifacts"] if entry.get("role") == MEMBER_ROLE]
    try:
        entries.sort(key=lambda entry: int(entry["member"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelLoadError("Model card has an ensemble file without a member number.") from exc
    names = [entry["file"] for entry in entries]
    info = card.get("model", {}).get("ensemble") if isinstance(card.get("model"), dict) else None
    if not isinstance(info, dict) or info.get("size") != len(names) or len(names) < MIN_MEMBERS:
        raise ModelLoadError("Model card ensemble does not match the files it lists.")
    if len(set(names)) != len(names):
        raise ModelLoadError("Model card lists the same ensemble file twice.")
    return names


def sd_ddof(card: Mapping) -> int:
    """Degrees-of-freedom correction the card says the reported SD uses (0 or 1)."""
    value = card["model"]["ensemble"].get("sd_ddof")
    if value not in (0, 1):
        raise ModelLoadError("Model card has an invalid SD setting.")
    return int(value)


class TargetEnsemble:
    """The loaded members for one target."""

    def __init__(self, members: Sequence, ddof: int):
        self.members = list(members)
        self.ddof = ddof

    @classmethod
    def load(cls, paths: Sequence, ddof: int) -> "TargetEnsemble":
        """Load verified TorchScript files (a file torch cannot read raises ModelLoadError)."""
        members = []
        for path in paths:
            try:
                module = torch.jit.load(str(path), map_location="cpu")
            except (RuntimeError, OSError, ValueError) as exc:
                raise ModelLoadError("Model file could not be loaded.") from exc
            module.eval()
            members.append(module)
        return cls(members, ddof)

    def predict(self, mol_graph: graph.MolGraph) -> Tuple[float, float, List[float]]:
        """(mean, SD, the members' own values) for one molecule graph."""
        x, edge_index, edge_attr, batch = graph.batch_graphs([mol_graph])
        tensors = (torch.from_numpy(x), torch.from_numpy(edge_index), torch.from_numpy(edge_attr),
                   torch.from_numpy(batch))
        with torch.inference_mode():
            values = [float(member(*tensors)[0]) for member in self.members]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("A model returned a value that is not a number.")
        return float(np.mean(values)), float(np.std(values, ddof=self.ddof)), values
