# QML result payload (MSP-27, extended by MSP-28)

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

## Real backend (MSP-28, `QML_BACKEND=vqe`)

The stub stays the default. With `QML_BACKEND=vqe` (PennyLane and PySCF installed from
`requirements-qml.txt`, Linux only) the same payload is returned with these differences. The four
agreed blocks keep their names; nothing is renamed.

| Field | Stub | Real backend |
|---|---|---|
| `is_stub`, `backend.is_stub` | `true` | `false` |
| `electronic_stability.metric` | `homo_lumo_gap` | `vqe_excitation_gap` (energy to the lowest excited singlet in the active space) |
| `electronic_stability.value` | fixed 4.5 eV | the VQE gap in eV, different for every molecule |
| `electronic_stability.limitation` | starts with `STUB:` | the real wording (`labels.LIMITATION_QML`) |
| `electronic_stability.warning` | never | present when not converged |
| `convergence.error_vs_fci_mha` | absent | larger of the ground- and excited-state errors against full CI (mHa) |
| `convergence.converged` | last change <= threshold | `error_vs_fci_mha` <= `threshold_mha` |
| `excluded_from_triage` | absent | `true` when not converged |
| `classical_references` | absent | Hartree-Fock gap and full-CI gap and energies, labelled |
| `run` | absent | mapping, ansatz, optimiser, simulator, basis, qubits, geometry, library, seconds |
| `energies` | absent | ground-state iteration energies and errors against full CI, excited-state energy and error, gap error |

**For the aggregator (MSP-31):** leave the QML block out of triage when `convergence.converged` is
false (`excluded_from_triage` says the same thing in one field), and when `is_stub` is true. A
non-converged result still has its value; the screen shows it with `electronic_stability.warning`.

**What the numbers are:** a simulated quantum algorithm (UCCSD ansatz, L-BFGS-B, exact statevector,
no noise) on 4 electrons in 4 orbitals (8 qubits, minimal STO-3G basis) around the Fermi level, on a
3D geometry from RDKit (ETKDG seed 7 + MMFF). `vqe_excitation_gap` is not the Hartree-Fock
HOMO-LUMO gap (`classical_references.hf_gap_ev`, shown beside it); it is not a measurement and is
not comparable with experiment. Both energies are compared with full CI of the same active space
from PySCF. H2 uses its 2 electrons in 2 orbitals (4 qubits), LiH 4 in 4.

```json
{
  "tier": "qml", "is_stub": false,
  "electronic_stability": {"metric": "vqe_excitation_gap", "value": 13.271136, "unit": "eV",
                           "limitation": "Estimated by a quantum-chemistry calculation ...",
                           "model_version": "qml_vqe@1.0.0"},
  "convergence": {"converged": true, "iterations": 5, "final_change_mha": 6e-06,
                  "threshold_mha": 1.6, "error_vs_fci_mha": 0.000188},
  "backend": {"name": "vqe", "version": "1.0.0", "is_stub": false},
  "active_space": {"electrons": 4, "orbitals": 4},
  "excluded_from_triage": false,
  "classical_references": {"hf_gap_ev": 25.30510216, "fci_gap_ev": 13.27113558, "hf_energy_ha": -113.54702328,
                           "fci_ground_energy_ha": -113.5506651, "fci_excited_energy_ha": -113.06295987,
                           "source": "PySCF 2.14.0, ...", "unit": "eV", "limitation": "...",
                           "model_version": "qml_vqe@1.0.0"},
  "run": {"mapping": "jordan_wigner", "ansatz": "UCCSD (from the Hartree-Fock state)", "optimiser": "L-BFGS-B",
          "simulator": "PennyLane default.qubit, exact statevector", "basis": "sto-3g", "qubits": 8,
          "geometry": "RDKit ETKDG (seed 7) + MMFF", "library": "PennyLane 0.45.1", "seconds": 11.7,
          "excited_state_method": "qEOM-style subspace expansion ..."},
  "energies": {"ground_state": {"energy_ha": -113.55066492, "error_vs_fci_mha": 0.000188, "iterations": 5,
                                "iteration_energies_ha": ["..."], "iteration_errors_vs_fci_mha": ["..."]},
               "excited_state": {"energy_ha": -113.06295987, "error_vs_fci_mha": 0.0,
                                 "subspace_ground_energy_ha": -113.5506651, "subspace_dimension": 20},
               "gap_error_ev": 0.0}
}
```

(Methanol, `CO`, run on 31 Oct 2026. Long texts shortened here.)
