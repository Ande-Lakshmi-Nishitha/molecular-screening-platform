"""PubChem client (MSP-21; FR-005, FR-055). The 5 requests/second cap is in limiter.py."""
from urllib.parse import quote

from backend.services.search_client.upstream import MAX_RESULTS, get_json, to_float

SERVICE = "PubChem"
BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PROPS = "CanonicalSMILES,IsomericSMILES,InChI,InChIKey,MolecularFormula,MolecularWeight,IUPACName"


def search_pubchem(query):
    """Return up to MAX_RESULTS normalised hits for a compound name (may be empty)."""
    url = f"{BASE}/compound/name/{quote(query, safe='')}/property/{PROPS}/JSON"
    data = get_json(SERVICE, url)
    props = ((data or {}).get("PropertyTable") or {}).get("Properties") or []
    results = []
    for p in props[:MAX_RESULTS]:
        cid = p.get("CID")
        results.append({
            "source": "pubchem",
            "id": str(cid) if cid is not None else None,
            "name": p.get("IUPACName"),
            # PubChem has renamed its SMILES fields over time, so accept each variant.
            "smiles": p.get("IsomericSMILES") or p.get("SMILES")
                      or p.get("CanonicalSMILES") or p.get("ConnectivitySMILES"),
            "inchi": p.get("InChI"),
            "inchi_key": p.get("InChIKey"),
            "formula": p.get("MolecularFormula"),
            "molecular_weight": to_float(p.get("MolecularWeight")),
            "structure_image_url": f"{BASE}/compound/cid/{cid}/PNG" if cid is not None else None,
        })
    return results
