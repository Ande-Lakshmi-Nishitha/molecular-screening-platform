"""All disclaimer, limitation and label wording in one file (minimal first version, MSP-23; DL added in MSP-26).

Owner: Nishitha. Requirements: NFR-007..011, BR-006, BR-007.
MSP-32 completes this file (QML wording for a real backend, all_wording(), forbidden_claims_in()) and adds
the schema test over every tier (tests/unit/test_result_schema.py).

Rules for every sentence kept here:
* Nothing says a molecule is safe, non-toxic, effective or approved (NFR-011).
* The Tox21 limitation names what Tox21 does not cover: hERG, hepatotoxicity and
  mutagenicity (NFR-009).
* A result block carries `unit`, `limitation` and `model_version` (NFR-007, NFR-008).
  Every Tox21 value carries the label `assay_signal` (BR-006).

Code builds result blocks from the helpers below and never writes its own wording.
Pure Python: no Flask, RDKit, database or Celery imports.
"""
import re
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


UNIT_EV = "eV"

LIMITATION_QML_STUB = (
    "STUB: a fixed demonstration value from a placeholder backend. No quantum calculation was "
    "run, and the same number is returned for every molecule, so it says nothing about this "
    "molecule's electronic stability and must not be used to compare or rank molecules. A real "
    "estimate of the HOMO-LUMO gap (in eV) replaces it in a later release."
)


def qml_stub_labels(model_version: str) -> Dict[str, str]:
    """Label fields for the electronic-stability block while the stub backend is in use."""
    return {"unit": UNIT_EV, "limitation": LIMITATION_QML_STUB, "model_version": model_version}


UNIT_ALERT_COUNT = "number of matching structural alerts"
UNIT_RULE_VIOLATIONS = "number of rule violations"
UNIT_TANIMOTO = "Tanimoto similarity (0 to 1)"

LIMITATION_ALERTS = (
    "Substructure patterns from the public PAINS and Brenk lists that are often linked to "
    "assay interference or to unwanted reactivity. A match is a reason to look closer, not a "
    "finding about this molecule; having no match does not mean there is no concern."
)

LIMITATION_RULES = (
    "Simple property rules of thumb (Lipinski rule of five, Veber). They describe how a "
    "molecule's size and polarity compare with the usual ranges; they do not predict "
    "absorption, activity or toxicity, and many useful molecules break them."
)

LIMITATION_DOMAIN = (
    "Compares this molecule with the molecules the models were trained on: the highest Tanimoto "
    "similarity to any training molecule, and whether key properties lie inside the training "
    "range. The limits are provisional and not tuned on measured results. Being inside the "
    "domain does not make a prediction correct."
)

WARNING_OUT_OF_DOMAIN = (
    "This molecule is unlike the molecules the models were trained on. Treat these predictions "
    "with extra caution."
)

WARNING_DOMAIN_UNAVAILABLE = (
    "The applicability domain could not be checked for this molecule. Treat these predictions "
    "with caution."
)


def alerts_labels(model_version: str) -> Dict[str, str]:
    """Label fields for the structural-alerts block."""
    return {"unit": UNIT_ALERT_COUNT, "limitation": LIMITATION_ALERTS, "model_version": model_version}


def rules_labels(model_version: str) -> Dict[str, str]:
    """Label fields for the Lipinski / Veber block."""
    return {"unit": UNIT_RULE_VIOLATIONS, "limitation": LIMITATION_RULES, "model_version": model_version}


def domain_labels(model_version: str) -> Dict[str, str]:
    """Label fields for the applicability-domain block."""
    return {"unit": UNIT_TANIMOTO, "limitation": LIMITATION_DOMAIN, "model_version": model_version}


LIMITATION_QML = (
    "Estimated by a quantum-chemistry calculation (a simulated quantum algorithm, not a measurement) for "
    "a small molecule in a small active space of a minimal basis set. The value is the energy needed to "
    "reach the lowest excited singlet state, an indicator of electronic stability only. It depends on the "
    "method, geometry and active space used, it is not the same as the Hartree-Fock HOMO-LUMO gap shown "
    "beside it, and it says nothing about reactivity, shelf life or biological behaviour. A run that did "
    "not converge is flagged and left out of the triage."
)

LIMITATION_QML_REFERENCE = (
    "Classical reference calculations on the same geometry, basis set and active space as the quantum "
    "run, shown so that the run can be judged. They are calculated values, not measurements. The "
    "Hartree-Fock gap is the lowest unoccupied minus the highest occupied orbital energy of the whole "
    "molecule; the full-CI gap is the exact energy to the lowest excited singlet inside the active space."
)

WARNING_QML_NOT_CONVERGED = (
    "This quantum calculation did not reach its accuracy target against the full configuration-"
    "interaction reference, so its value is less reliable and is left out of the triage."
)

# Words that must never be used to describe a molecule (NFR-011), matched as whole words.
FORBIDDEN_CLAIMS = ("safe", "non-toxic", "nontoxic", "harmless", "effective", "approved", "cure", "cured")


def qml_labels(is_stub: bool, model_version: str) -> Dict[str, str]:
    """Label fields for the electronic-stability block: the stub wording or the real-backend wording."""
    if is_stub:
        return qml_stub_labels(model_version)
    return {"unit": UNIT_EV, "limitation": LIMITATION_QML, "model_version": model_version}


def qml_reference_labels(model_version: str) -> Dict[str, str]:
    """Label fields for the classical-reference block that sits beside a real quantum result."""
    return {"unit": UNIT_EV, "limitation": LIMITATION_QML_REFERENCE, "model_version": model_version}


def all_wording() -> Dict[str, str]:
    """Every fixed sentence in this file by constant name (the schema test checks them all)."""
    return {name: value for name, value in sorted(globals().items())
            if isinstance(value, str)
            and name.split("_")[0] in ("LIMITATION", "WARNING", "DISCLAIMER")}


def forbidden_claims_in(text: str) -> List[str]:
    """The forbidden whole words found in a text (empty when the text is clean)."""
    lowered = text.lower()
    return [word for word in FORBIDDEN_CLAIMS
            if re.search(r"(?<![a-z0-9])" + re.escape(word) + r"(?![a-z0-9])", lowered)]
