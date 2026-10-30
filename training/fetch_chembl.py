"""Download and clean ChEMBL activity data for the DL targets (MSP-25).

Run from the repository root (needs internet access to www.ebi.ac.uk):

  python -m training.fetch_chembl                  # every target in backend/config/targets.yaml
  python -m training.fetch_chembl --target EGFR    # one target

For each target it writes, into training/data/ (git-ignored):
  chembl_<CHEMBL id>_raw.jsonl   the activity records that were downloaded (kept for audit)
  chembl_<CHEMBL id>.csv         cleaned table: smiles_std, inchikey, pactivity, n_measurements
  chembl_<CHEMBL id>.meta.json   query, date, ChEMBL release, row counts, SHA-256 of the raw file

Cleaning rules (every step is counted in the meta file, nothing is dropped silently):
  * binding assays only (assay_type B), exact values only (relation "="), activity type from the
    target's unit (pIC50 -> IC50, pKi -> Ki), pChEMBL value present
  * records ChEMBL flags as invalid (data_validity_comment) or as potential duplicates are dropped
  * structures go through the MSP-19 standardiser, the same one the API uses
  * only molecules the DL tier can score: heavy atoms limited to the graph featuriser's elements
  * repeated measurements of one molecule (same InChIKey) become their median; molecules whose
    repeats disagree by more than MAX_SPREAD log units are dropped as unreliable

Licence: ChEMBL is CC BY-SA 3.0. Record the licence from the ChEMBL download page on the download day (BR-009).
"""
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from backend.core.config import load_yaml_safe
from backend.workers.dl import graph
from training.datasets import standardise_many
from training.utils import file_sha256, utc_now_iso, write_json

CHEMBL_HOST = "https://www.ebi.ac.uk"
ACTIVITY_URL = CHEMBL_HOST + "/chembl/api/data/activity.json"
STATUS_URL = CHEMBL_HOST + "/chembl/api/data/status.json"
ACTIVITY_TYPE = {"pIC50": "IC50", "pKi": "Ki"}
PAGE_SIZE = 1000
MAX_SPREAD = 2.0           # log units; repeats that differ more than this are dropped
PACTIVITY_RANGE = (2.0, 12.0)
MIN_ROWS = 200             # fewer clean molecules than this: stop and tell the user
KEPT_FIELDS = (
    "molecule_chembl_id", "canonical_smiles", "pchembl_value", "standard_type", "standard_relation",
    "assay_type", "data_validity_comment", "potential_duplicate", "target_chembl_id", "assay_chembl_id",
)
LICENCE_NOTE = "ChEMBL is licensed CC BY-SA 3.0. Verify on the ChEMBL download page and record the date."


class DataError(RuntimeError):
    """The download or the cleaned data cannot be used."""


def make_session() -> requests.Session:
    """A requests session that retries rate limits and server errors with a back-off."""
    session = requests.Session()
    retry = Retry(total=5, backoff_factor=2.0, status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=("GET",))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers["User-Agent"] = "msp-training/1.0 (university project; contact: repository owner)"
    return session


def target_list(path, only: Optional[str] = None) -> List[Dict]:
    """Targets from targets.yaml (all of them, or just the one whose id or CHEMBL id matches)."""
    data = load_yaml_safe(path) or {}
    targets = data.get("targets") or []
    for target in targets:
        for key in ("id", "chembl_id", "unit"):
            if not target.get(key):
                raise DataError(f"targets.yaml: an entry has no '{key}'")
        if target["unit"] not in ACTIVITY_TYPE:
            raise DataError(f"targets.yaml: unit of {target['id']} must be one of {sorted(ACTIVITY_TYPE)}")
    if only:
        targets = [t for t in targets if only in (t["id"], t["chembl_id"])]
        if not targets:
            raise DataError(f"no target '{only}' in targets.yaml")
    return targets


def fetch_records(chembl_id: str, activity_type: str, session, pause: float = 0.2,
                  max_pages: int = 500) -> List[Dict]:  # pylint: disable=too-many-arguments,too-many-positional-arguments
    """All activity records with a pChEMBL value for one target and activity type."""
    params = {
        "target_chembl_id": chembl_id,
        "standard_type": activity_type,
        "pchembl_value__isnull": "false",
        "limit": PAGE_SIZE,
        "format": "json",
    }
    url, records = ACTIVITY_URL, []
    for page in range(max_pages):
        response = session.get(url, params=params if page == 0 else None, timeout=(10, 120))
        if response.status_code != 200:
            raise DataError(f"ChEMBL answered HTTP {response.status_code} for {chembl_id}")
        body = response.json()
        for item in body.get("activities", []):
            records.append({field: item.get(field) for field in KEPT_FIELDS})
        following = (body.get("page_meta") or {}).get("next")
        print(f"  {chembl_id} {activity_type}: page {page + 1}, {len(records)} records", flush=True)
        if not following:
            return records
        url = following if following.startswith("http") else CHEMBL_HOST + following
        time.sleep(pause)
    raise DataError(f"more than {max_pages} pages for {chembl_id}; stopped")


def chembl_release(session) -> Optional[str]:
    """ChEMBL release name (for the meta file); None if the status call fails."""
    try:
        return session.get(STATUS_URL, params={"format": "json"}, timeout=(10, 30)).json().get("chembl_db_version")
    except (requests.RequestException, ValueError):
        return None


def write_raw(records: Iterable[Dict], path: Path) -> str:
    """Write records as JSON lines and return the file's SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return file_sha256(path)


def read_raw(path: Path) -> List[Dict]:
    """Read a JSON-lines file written by write_raw."""
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _usable(record: Dict, activity_type: str) -> bool:
    if record.get("standard_type") != activity_type or record.get("standard_relation") != "=":
        return False
    if record.get("assay_type") != "B" or record.get("data_validity_comment"):
        return False
    if str(record.get("potential_duplicate")) in ("1", "True", "true"):
        return False
    return bool(record.get("canonical_smiles")) and record.get("pchembl_value") not in (None, "")


def _supported(smiles: str) -> bool:
    try:
        graph.smiles_to_graph(smiles)
    except graph.GraphError:
        return False
    return True


def curate(records: List[Dict], activity_type: str, max_spread: float = MAX_SPREAD):
    """Clean the downloaded records. Returns (table, stats)."""
    stats: Dict[str, int] = {
        "n_downloaded": len(records), "n_dropped_filters": 0, "n_dropped_out_of_range": 0,
        "n_dropped_unparsable": 0, "n_dropped_unsupported_atoms": 0, "n_dropped_inconsistent_repeats": 0,
        "n_measurements_used": 0, "n_clean": 0,
    }
    kept = [record for record in records if _usable(record, activity_type)]
    stats["n_dropped_filters"] = len(records) - len(kept)
    frame = pd.DataFrame({
        "smiles": [record["canonical_smiles"] for record in kept],
        "pactivity": [float(record["pchembl_value"]) for record in kept],
    })
    in_range = frame["pactivity"].between(*PACTIVITY_RANGE)
    stats["n_dropped_out_of_range"] = int((~in_range).sum())
    frame = frame[in_range].reset_index(drop=True)
    if frame.empty:
        return pd.DataFrame(columns=["smiles_std", "inchikey", "pactivity", "n_measurements"]), stats

    std = standardise_many(frame["smiles"].tolist())
    joined = pd.concat([frame, std], axis=1)
    stats["n_dropped_unparsable"] = int((~joined["ok"]).sum())
    joined = joined[joined["ok"]]
    supported = joined["smiles_std"].map(_supported)
    stats["n_dropped_unsupported_atoms"] = int((~supported).sum())
    joined = joined[supported]

    grouped = joined.groupby("inchikey", sort=True)["pactivity"].agg(["median", "min", "max", "count"])
    reliable = (grouped["max"] - grouped["min"]) <= max_spread
    stats["n_dropped_inconsistent_repeats"] = int((~reliable).sum())
    grouped = grouped[reliable]
    first_smiles = joined.groupby("inchikey", sort=True)["smiles_std"].first()
    table = pd.DataFrame({
        "smiles_std": first_smiles.loc[grouped.index].to_numpy(),
        "inchikey": grouped.index.to_numpy(),
        "pactivity": grouped["median"].to_numpy(),
        "n_measurements": grouped["count"].to_numpy(),
    }).reset_index(drop=True)
    stats["n_measurements_used"] = int(grouped["count"].sum())
    stats["n_clean"] = int(len(table))
    return table, stats


def run_target(target: Dict, out_dir: Path, session, refresh: bool = False,
               min_rows: int = MIN_ROWS) -> Dict:  # pylint: disable=too-many-arguments,too-many-positional-arguments
    """Download (or reuse) and clean the data for one target; write the csv and meta files."""
    chembl_id = target["chembl_id"]
    activity_type = ACTIVITY_TYPE[target["unit"]]
    raw_path = out_dir / f"chembl_{chembl_id}_raw.jsonl"
    if raw_path.exists() and not refresh:
        print(f"  reusing {raw_path.name} (use --refresh to download again)", flush=True)
        records = read_raw(raw_path)
        raw_sha = file_sha256(raw_path)
    else:
        records = fetch_records(chembl_id, activity_type, session)
        raw_sha = write_raw(records, raw_path)
    table, stats = curate(records, activity_type)
    if len(table) < min_rows:
        raise DataError(f"{target['id']}: only {len(table)} clean molecules (need {min_rows}); see {raw_path}")
    csv_path = out_dir / f"chembl_{chembl_id}.csv"
    table.to_csv(csv_path, index=False, lineterminator="\n")
    meta = {
        "target_id": target["id"],
        "chembl_id": chembl_id,
        "unit": target["unit"],
        "activity_type": activity_type,
        "source": "ChEMBL",
        "origin": ACTIVITY_URL,
        "query": {"target_chembl_id": chembl_id, "standard_type": activity_type, "pchembl_value__isnull": "false"},
        "chembl_release": chembl_release(session) if session is not None else None,
        "retrieved_at": utc_now_iso(),
        "licence_note": LICENCE_NOTE,
        "licence_verified": False,
        "raw_file": raw_path.name,
        "raw_sha256": raw_sha,
        "csv_file": csv_path.name,
        "csv_sha256": file_sha256(csv_path),
        "max_spread": MAX_SPREAD,
        "pactivity_range": list(PACTIVITY_RANGE),
        **stats,
    }
    write_json(out_dir / f"chembl_{chembl_id}.meta.json", meta)
    return meta


def main(argv=None) -> None:  # pragma: no cover - command line wrapper
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--targets-file", type=Path, default=Path("backend/config/targets.yaml"))
    parser.add_argument("--target", help="only this target (id or CHEMBL id)")
    parser.add_argument("--out", type=Path, default=Path("training/data"))
    parser.add_argument("--refresh", action="store_true", help="download again even if the raw file exists")
    args = parser.parse_args(argv)
    session = make_session()
    for target in target_list(args.targets_file, args.target):
        print(f"{target['id']} ({target['chembl_id']}, {target['unit']})", flush=True)
        meta = run_target(target, args.out, session, args.refresh)
        counts = {key: value for key, value in meta.items() if key.startswith("n_")}
        print("  done:", counts, flush=True)
    print("All targets done. Files are in", args.out)


if __name__ == "__main__":  # pragma: no cover
    main()
