"""Shared molecule featuriser for the ML tier (MSP-22 training, MSP-23 serving).

One function turns a standardised RDKit molecule into one fixed-length vector:
RDKit 2D descriptors, then a 2048-bit Morgan fingerprint, then MACCS keys
(FR-009). Training and serving both call featurise_mol(), so a model can never
see features built a different way.

Pure RDKit and numpy: no Flask, database or Celery imports. The molecule must
already be standardised (MSP-19); this module does not standardise.

The descriptor list is frozen by name on purpose. RDKit can add or remove
descriptors between releases, which would silently shift every column. A unit
test fails if a frozen name disappears, and every model card records
feature_signature() so serving can refuse a model built with other features.
"""
# RDKit is a C extension, so pylint cannot see its members.
# pylint: disable=no-member,no-name-in-module,c-extension-no-member
import hashlib
from typing import Iterable, List

import numpy as np
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator

FEATURISER_VERSION = "1"
MORGAN_RADIUS = 2
MORGAN_BITS = 2048
MACCS_BITS = 167
# Descriptor values are clipped to this range so float32 never overflows.
VALUE_LIMIT = 1e12

# RDKit descriptors, frozen by name (Ipc is left out: its values grow
# astronomically with molecule size and carry no usable signal).
DESCRIPTOR_NAMES = (
    "MaxAbsEStateIndex", "MaxEStateIndex", "MinAbsEStateIndex", "MinEStateIndex", "qed",
    "SPS", "MolWt", "HeavyAtomMolWt", "ExactMolWt", "NumValenceElectrons",
    "NumRadicalElectrons", "MaxPartialCharge", "MinPartialCharge", "MaxAbsPartialCharge",
    "MinAbsPartialCharge", "FpDensityMorgan1", "FpDensityMorgan2", "FpDensityMorgan3",
    "BCUT2D_MWHI", "BCUT2D_MWLOW", "BCUT2D_CHGHI", "BCUT2D_CHGLO", "BCUT2D_LOGPHI",
    "BCUT2D_LOGPLOW", "BCUT2D_MRHI", "BCUT2D_MRLOW", "AvgIpc", "BalabanJ", "BertzCT",
    "Chi0", "Chi0n", "Chi0v", "Chi1", "Chi1n", "Chi1v", "Chi2n", "Chi2v", "Chi3n",
    "Chi3v", "Chi4n", "Chi4v", "HallKierAlpha", "Kappa1", "Kappa2", "Kappa3", "LabuteASA",
    "PEOE_VSA1", "PEOE_VSA10", "PEOE_VSA11", "PEOE_VSA12", "PEOE_VSA13", "PEOE_VSA14",
    "PEOE_VSA2", "PEOE_VSA3", "PEOE_VSA4", "PEOE_VSA5", "PEOE_VSA6", "PEOE_VSA7",
    "PEOE_VSA8", "PEOE_VSA9", "SMR_VSA1", "SMR_VSA10", "SMR_VSA2", "SMR_VSA3", "SMR_VSA4",
    "SMR_VSA5", "SMR_VSA6", "SMR_VSA7", "SMR_VSA8", "SMR_VSA9", "SlogP_VSA1",
    "SlogP_VSA10", "SlogP_VSA11", "SlogP_VSA12", "SlogP_VSA2", "SlogP_VSA3", "SlogP_VSA4",
    "SlogP_VSA5", "SlogP_VSA6", "SlogP_VSA7", "SlogP_VSA8", "SlogP_VSA9", "TPSA",
    "EState_VSA1", "EState_VSA10", "EState_VSA11", "EState_VSA2", "EState_VSA3",
    "EState_VSA4", "EState_VSA5", "EState_VSA6", "EState_VSA7", "EState_VSA8",
    "EState_VSA9", "VSA_EState1", "VSA_EState10", "VSA_EState2", "VSA_EState3",
    "VSA_EState4", "VSA_EState5", "VSA_EState6", "VSA_EState7", "VSA_EState8",
    "VSA_EState9", "FractionCSP3", "HeavyAtomCount", "NHOHCount", "NOCount",
    "NumAliphaticCarbocycles", "NumAliphaticHeterocycles", "NumAliphaticRings",
    "NumAmideBonds", "NumAromaticCarbocycles", "NumAromaticHeterocycles",
    "NumAromaticRings", "NumAtomStereoCenters", "NumBridgeheadAtoms", "NumHAcceptors",
    "NumHDonors", "NumHeteroatoms", "NumHeterocycles", "NumRotatableBonds",
    "NumSaturatedCarbocycles", "NumSaturatedHeterocycles", "NumSaturatedRings",
    "NumSpiroAtoms", "NumUnspecifiedAtomStereoCenters", "Phi", "RingCount", "MolLogP",
    "MolMR", "fr_Al_COO", "fr_Al_OH", "fr_Al_OH_noTert", "fr_ArN", "fr_Ar_COO", "fr_Ar_N",
    "fr_Ar_NH", "fr_Ar_OH", "fr_COO", "fr_COO2", "fr_C_O", "fr_C_O_noCOO", "fr_C_S",
    "fr_HOCCN", "fr_Imine", "fr_NH0", "fr_NH1", "fr_NH2", "fr_N_O", "fr_Ndealkylation1",
    "fr_Ndealkylation2", "fr_Nhpyrrole", "fr_SH", "fr_aldehyde", "fr_alkyl_carbamate",
    "fr_alkyl_halide", "fr_allylic_oxid", "fr_amide", "fr_amidine", "fr_aniline",
    "fr_aryl_methyl", "fr_azide", "fr_azo", "fr_barbitur", "fr_benzene",
    "fr_benzodiazepine", "fr_bicyclic", "fr_diazo", "fr_dihydropyridine", "fr_epoxide",
    "fr_ester", "fr_ether", "fr_furan", "fr_guanido", "fr_halogen", "fr_hdrzine",
    "fr_hdrzone", "fr_imidazole", "fr_imide", "fr_isocyan", "fr_isothiocyan", "fr_ketone",
    "fr_ketone_Topliss", "fr_lactam", "fr_lactone", "fr_methoxy", "fr_morpholine",
    "fr_nitrile", "fr_nitro", "fr_nitro_arom", "fr_nitro_arom_nonortho", "fr_nitroso",
    "fr_oxazole", "fr_oxime", "fr_para_hydroxylation", "fr_phenol",
    "fr_phenol_noOrthoHbond", "fr_phos_acid", "fr_phos_ester", "fr_piperdine",
    "fr_piperzine", "fr_priamide", "fr_prisulfonamd", "fr_pyridine", "fr_quatN",
    "fr_sulfide", "fr_sulfonamd", "fr_sulfone", "fr_term_acetylene", "fr_tetrazole",
    "fr_thiazole", "fr_thiocyan", "fr_thiophene", "fr_unbrch_alkane", "fr_urea",
)

N_DESCRIPTORS = len(DESCRIPTOR_NAMES)
FEATURE_NAMES = (
    list(DESCRIPTOR_NAMES)
    + [f"morgan_{i}" for i in range(MORGAN_BITS)]
    + [f"maccs_{i}" for i in range(MACCS_BITS)]
)
N_FEATURES = len(FEATURE_NAMES)
MORGAN_SLICE = slice(N_DESCRIPTORS, N_DESCRIPTORS + MORGAN_BITS)

_DESCRIPTOR_FUNCTIONS = None
_MORGAN_GENERATOR = None


def _descriptor_functions():
    """Descriptor callables in frozen order (looked up once, on first use)."""
    global _DESCRIPTOR_FUNCTIONS  # pylint: disable=global-statement
    if _DESCRIPTOR_FUNCTIONS is None:
        available = dict(Descriptors.descList)
        missing = [name for name in DESCRIPTOR_NAMES if name not in available]
        if missing:
            raise RuntimeError(
                "This RDKit version no longer has descriptors: " + ", ".join(missing)
            )
        _DESCRIPTOR_FUNCTIONS = [available[name] for name in DESCRIPTOR_NAMES]
    return _DESCRIPTOR_FUNCTIONS


def _morgan_generator():
    global _MORGAN_GENERATOR  # pylint: disable=global-statement
    if _MORGAN_GENERATOR is None:
        _MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
            radius=MORGAN_RADIUS, fpSize=MORGAN_BITS
        )
    return _MORGAN_GENERATOR


def descriptor_values(mol) -> np.ndarray:
    """The frozen RDKit descriptors as float64; a failed descriptor is NaN."""
    values = np.full(N_DESCRIPTORS, np.nan, dtype=np.float64)
    for index, function in enumerate(_descriptor_functions()):
        try:
            values[index] = function(mol)
        except Exception:  # pylint: disable=broad-exception-caught
            values[index] = np.nan  # one bad descriptor must not lose the molecule
    values[~np.isfinite(values)] = np.nan
    return np.clip(values, -VALUE_LIMIT, VALUE_LIMIT)


def morgan_bits(mol) -> np.ndarray:
    """2048-bit Morgan fingerprint (radius 2) as uint8 zeros and ones."""
    return _morgan_generator().GetFingerprintAsNumPy(mol).astype(np.uint8)


def maccs_bits(mol) -> np.ndarray:
    """167 MACCS keys as uint8 zeros and ones (bit 0 is unused by RDKit)."""
    bits = np.zeros(MACCS_BITS, dtype=np.uint8)
    bits[list(MACCSkeys.GenMACCSKeys(mol).GetOnBits())] = 1
    return bits


def featurise_mol(mol) -> np.ndarray:
    """Feature vector for one standardised molecule: float32, length N_FEATURES.

    Missing descriptor values are NaN, which XGBoost treats as missing.
    """
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("featurise_mol needs a valid, non-empty molecule")
    vector = np.concatenate(
        [descriptor_values(mol), morgan_bits(mol), maccs_bits(mol)]
    )
    return vector.astype(np.float32)


def featurise_mols(mols: Iterable) -> np.ndarray:
    """Stack of feature vectors, shape (n_molecules, N_FEATURES)."""
    rows: List[np.ndarray] = [featurise_mol(mol) for mol in mols]
    if not rows:
        return np.zeros((0, N_FEATURES), dtype=np.float32)
    return np.vstack(rows)


def feature_signature() -> str:
    """Short hash of the featuriser version and column names, stored in cards."""
    text = FEATURISER_VERSION + "|" + ",".join(FEATURE_NAMES)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
