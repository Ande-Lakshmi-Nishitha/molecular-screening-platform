"""Golden molecule set (SRS 6.2): used by unit, integration and smoke tests."""

GOLDEN = {
    "aspirin": {
        "smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "inchikey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
        "formula": "C9H8O4",
        "heavy_atoms": 13,
        "qml_eligible": False,
    },
    "caffeine": {
        "smiles": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
        "inchikey": "RYYVLZVUVIJVGH-UHFFFAOYSA-N",
        "formula": "C8H10N4O2",
        "heavy_atoms": 14,
        "qml_eligible": False,
    },
    "ibuprofen": {
        "smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "inchikey": "HEFNNWSXXWATRW-UHFFFAOYSA-N",
        "formula": "C13H18O2",
        "heavy_atoms": 15,
        "qml_eligible": False,
    },
    "ethanol": {
        "smiles": "CCO",
        "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        "formula": "C2H6O",
        "heavy_atoms": 3,
        "qml_eligible": True,
    },
    "acetamide": {
        "smiles": "CC(N)=O",
        "inchikey": "DLFVBJFMPXGRIB-UHFFFAOYSA-N",
        "formula": "C2H5NO",
        "heavy_atoms": 4,
        "qml_eligible": True,
    },
}
