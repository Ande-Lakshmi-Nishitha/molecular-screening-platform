# ML result payload: fields added by MSP-24

Owner: Nishitha. Stored in `results.payload` of the task whose tier is `ml`. These three blocks are
**added**; the existing fields (`tier`, `solubility`, `tox21`) are unchanged. Pinned by
`tests/unit/workers_ml/test_msp24.py`. Every block also carries `unit`, `limitation` and `model_version`.

```json
{
  "alerts": {"pains": ["ene_rhod_A(235)"], "brenk": ["Michael_acceptor_1", "Thiocarbonyl_group"],
             "n_pains": 1, "n_brenk": 2},
  "rules":  {"lipinski": {"values": {"mol_wt": 221.3, "log_p": 2.18, "h_bond_donors": 1, "h_bond_acceptors": 3},
                          "limits": {"mol_wt": 500.0, "log_p": 5.0, "h_bond_donors": 5, "h_bond_acceptors": 10},
                          "failed": [], "violations": 0},
             "veber":    {"values": {"rotatable_bonds": 2, "tpsa": 63.6},
                          "limits": {"rotatable_bonds": 10, "tpsa": 140.0}, "failed": [], "violations": 0}},
  "domain": {"in_domain": true,
             "warning": null,
             "families": {"solubility": {"available": true, "in_domain": true, "max_tanimoto": 0.62,
                                         "threshold": 0.3, "descriptors_outside_range": []},
                          "tox21":      {"available": true, "in_domain": true, "max_tanimoto": 0.62,
                                         "threshold": 0.3, "descriptors_outside_range": []}}}
}
```

* **alerts**: PAINS and Brenk pattern names from RDKit's filter catalogs, sorted. A match is a reason to look
  closer, not a finding; no match is not reassurance (wording in `core/labels.py`).
* **rules**: counts of rule-of-five and Veber violations with the values and the limits used.
* **domain.in_domain**: `true` when both model families see the molecule as inside the training domain, `false`
  when either does not, `null` when a family could not be checked (reference data missing or failed its checks;
  then `families.<name>.available` is false). `warning` holds the fixed sentence to show when it is `false` or `null`.
* A family is inside when the highest Tanimoto similarity to a training molecule is at least `threshold`
  (config `AD_TANIMOTO_MIN`, 0.3, provisional, TBD-C8) and none of MolWt, MolLogP, TPSA, NumHDonors, NumHAcceptors,
  NumRotatableBonds, HeavyAtomCount lies outside the training min..max (`descriptors_outside_range` lists them).
* The limits are provisional and were not tuned on measured errors. Being inside does not make a prediction correct.
* `results.params_hash` now also covers the RDKit version, the rule limits, the threshold and the reference files.
