"""ChEMBL client (MSP-21; FR-005)."""
from backend.services.search_client.upstream import MAX_RESULTS, get_json, to_float

SERVICE = "ChEMBL"
BASE = "https://www.ebi.ac.uk/chembl/api/data"


def search_chembl(query):
    """Return up to MAX_RESULTS normalised hits (may be empty)."""
    data = get_json(SERVICE, f"{BASE}/molecule/search.json", params={"q": query, "limit": MAX_RESULTS})
    results = []
    for m in ((data or {}).get("molecules") or [])[:MAX_RESULTS]:
        structures = m.get("molecule_structures") or {}
        props = m.get("molecule_properties") or {}
        chembl_id = m.get("molecule_chembl_id")
        results.append({
            "source": "chembl",
            "id": chembl_id,
            "name": m.get("pref_name"),
            "smiles": structures.get("canonical_smiles"),
            "inchi": structures.get("standard_inchi"),
            "inchi_key": structures.get("standard_inchi_key"),
            "formula": props.get("full_molformula"),
            "molecular_weight": to_float(props.get("full_mwt")),
            "structure_image_url": f"{BASE}/image/{chembl_id}.svg" if chembl_id else None,
        })
    return results
