# QML result payload (MSP-27)

Owner: Nishitha. Agreed with Ayushi for the aggregator and triage (MSP-31). Stored in
`results.payload` of the task whose tier is `qml`. A unit test pins these exact field names
(`tests/unit/workers_qml/test_qml.py`); change them only together with the aggregator.

```json
{
  "tier": "qml",
  "is_stub": true,
  "electronic_stability": {
    "metric": "homo_lumo_gap",
    "value": 4.5,
    "unit": "eV",
    "limitation": "STUB: a fixed demonstration value ...",
    "model_version": "qml_stub@1.0.0"
  },
  "convergence": {"converged": true, "iterations": 12, "final_change_mha": 0.9, "threshold_mha": 1.6},
  "backend": {"name": "stub", "version": "1.0.0", "is_stub": true},
  "active_space": {"electrons": 4, "orbitals": 4}
}
```

* `results.model_version` is `qml_<backend>@<version>`, here `qml_stub@1.0.0`.
* `electronic_stability.value` is the estimated HOMO-LUMO gap in eV.
* `convergence.converged` is false when the last energy change is above `threshold_mha`
  (config `QML_CONV_THRESHOLD_MHA`, 1.6 mHa). The value is still present; show a "not converged"
  warning with it.
* **Stub:** while `is_stub` is true the value is the same fixed number for every molecule and no
  quantum calculation ran. It is marked three ways: `is_stub`, `backend.is_stub` and a limitation
  that starts with `STUB:`. The aggregator must not use it to rank or decide; show it as illustrative.
* A molecule with more than 9 heavy atoms, or atoms other than C, N, O, F, gets no QML task
  (skip reason `NO_FRAGMENT`). A QML task that fails is stored like the other tiers
  (`MODEL_UNAVAILABLE`, `TIMEOUT`, `COMPUTE_ERROR`); no result row means the tier is unavailable and
  the task's error is the reason.
