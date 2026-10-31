"""Active-space integrals and classical references from PySCF (MSP-28; FR-024).

Owner: Nishitha. One Hartree-Fock run and one CASCI run give both what the quantum run needs (the
active-space integrals) and what it is judged against (the references), so the two cannot disagree about
geometry, basis set or active space. The result screen labels the references "classical".

  hf_gap_ev     lowest unoccupied minus highest occupied Hartree-Fock orbital energy of the whole molecule
  fci_*         full CI inside the active space, singlet states only: ground and first excited singlet

PySCF is imported inside the function: it is only installed for the QML worker (requirements-qml.txt)
and has no Windows build.
"""
from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np

HA_TO_EV = 27.211386245988
BASIS = "sto-3g"


@dataclass(frozen=True)
class ActiveSpace:
    """What PySCF gives for one molecule: the integrals of the active space and the classical references."""

    core_energy: float          # nuclear repulsion plus the frozen-core electrons
    one_body: np.ndarray        # h[p, q]
    two_body: np.ndarray        # (pq|rs), chemists' notation, full 4-index array
    references: Dict[str, float]


def active_space(symbols: Sequence[str], coords_angstrom: Sequence[Sequence[float]],
                 electrons: int, orbitals: int, charge: int = 0) -> ActiveSpace:
    """Hartree-Fock, then full CI of `electrons` electrons in `orbitals` orbitals around the Fermi level."""
    from pyscf import ao2mo, fci, gto, mcscf, scf  # pylint: disable=import-outside-toplevel,import-error
    mol = gto.M(atom=[[sym, tuple(float(c) for c in xyz)] for sym, xyz in zip(symbols, coords_angstrom)],
                basis=BASIS, unit="Angstrom", charge=charge, spin=0, verbose=0)
    mean_field = scf.RHF(mol)
    mean_field.run()
    if not mean_field.converged:
        raise RuntimeError("Hartree-Fock did not converge")
    occupied = int(mol.nelectron // 2)
    energies = mean_field.mo_energy
    cas = mcscf.CASCI(mean_field, orbitals, electrons)
    cas.fcisolver = fci.direct_spin0.FCI(mol)   # singlet states only; the default solver also returns triplets
    cas.fcisolver.nroots = 2
    cas.verbose = 0
    cas.kernel()
    ground, excited = (float(e) for e in sorted(cas.e_tot))
    one_body, core_energy = cas.get_h1eff()
    two_body = ao2mo.restore(1, cas.get_h2eff(), orbitals)
    references = {
        "hf_energy_ha": float(mean_field.e_tot),
        "hf_gap_ev": float((energies[occupied] - energies[occupied - 1]) * HA_TO_EV),
        "fci_ground_energy_ha": ground,
        "fci_excited_energy_ha": excited,
        "fci_gap_ev": (excited - ground) * HA_TO_EV,
    }
    return ActiveSpace(float(core_energy), np.array(one_body), np.array(two_body), references)
