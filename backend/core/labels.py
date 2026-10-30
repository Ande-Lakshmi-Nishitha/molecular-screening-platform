"""All disclaimer, limitation and label wording in one file (minimal first version, MSP-23; DL added in MSP-26).

Owner: Nishitha. Requirements: NFR-007..011, BR-006, BR-007.
MSP-32 completes this file (QML and DL wording, the schema test over every tier).

Rules for every sentence kept here:
* Nothing says a molecule is safe, non-toxic, effective or approved (NFR-011).
* The Tox21 limitation names what Tox21 does not cover: hERG, hepatotoxicity and
  mutagenicity (NFR-009).
* A result block carries `unit`, `limitation` and `model_version` (NFR-007, NFR-008).
  Every Tox21 value carries the label `assay_signal` (BR-006).

Code builds result blocks from the helpers below and never writes its own wording.
Pure Python: no Flask, RDKit, database or Celery imports.
"""
from typing import Dict, List, Mapping

# Fields every result block must hold. missing_labels() reports the ones that do not.
REQUIRED_BLOCK_FIELDS = ("unit", "limitation", "model_version")

ASSAY_SIGNAL = "assay_signal"

UNIT_LOGS = "log10(mol/L)"
UNIT_ASSAY_PROBABILITY = "calibrated probability (0 to 1)"

DISCLAIMER = (
    "These are preliminary computational estimates for research screening only. "
    "They are not experimental results and do not replace laboratory testing or expert review."
)

LIMITATION_SOLUBILITY = (
    "Estimated by a machine-learning model trained on public solubility data; an indicative "
    "value, not a measurement. The interval is a nominal 90% prediction interval and can be "
    "wrong, especially for molecules unlike the training data."
)

LIMITATION_TOX21 = (
    "Each value is a calibrated signal for one Tox21 high-throughput assay, not a toxicity "
    "assessment. Tox21 does not cover hERG, hepatotoxicity or mutagenicity. "
    "A low value is not evidence that a molecule is free of toxicity."
)


LIMITATION_ACTIVITY = (
    "Predicted by an ensemble of five graph neural networks trained on public ChEMBL binding "
    "data for this target. The value is an estimate, not a measurement, and says nothing about "
    "activity in cells, animals or patients. The standard deviation shows how much the five "
    "models disagree; it is not the real error and can be small for a wrong prediction, "
    "especially for molecules unlike the training data."
)

LIMITATION_BASELINE = (
    "A simpler gradient-boosted tree model on fingerprints and descriptors, trained on the same "
    "data. It is shown only for comparison with the network ensemble, and has the same limits."
)


def activity_labels(unit: str, model_version: str) -> Dict[str, str]:
    """Label fields for the DL activity block (the unit is the target's own: pIC50 or pKi)."""
    return {"unit": unit, "limitation": LIMITATION_ACTIVITY, "model_version": model_version}


def baseline_labels(unit: str, model_version: str) -> Dict[str, str]:
    """Label fields for the DL baseline block."""
    return {"unit": unit, "limitation": LIMITATION_BASELINE, "model_version": model_version}


def solubility_labels(model_version: str) -> Dict[str, str]:
    """Label fields for the solubility result block."""
    return {"unit": UNIT_LOGS, "limitation": LIMITATION_SOLUBILITY, "model_version": model_version}


def tox21_labels(model_version: str) -> Dict[str, str]:
    """Label fields for the Tox21 result block (each value also gets ASSAY_SIGNAL)."""
    return {
        "unit": UNIT_ASSAY_PROBABILITY,
        "limitation": LIMITATION_TOX21,
        "model_version": model_version,
        "label": ASSAY_SIGNAL,
    }


def missing_labels(block: Mapping) -> List[str]:
    """Names of the required label fields that are absent or empty in a result block."""
    return [name for name in REQUIRED_BLOCK_FIELDS if not block.get(name)]
