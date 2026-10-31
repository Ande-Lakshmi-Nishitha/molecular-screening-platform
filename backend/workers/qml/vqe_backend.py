"""Real quantum backend: VQE ground state plus first excited singlet by subspace expansion, on a statevector (MSP-28).

Owner: Nishitha. Requirements: FR-023..027, TBD-C5 (PennyLane), TBD-C6 (1.6 mHa, 4 electrons in 4 orbitals).

What it computes, for a closed-shell molecule of up to 9 heavy atoms:
  1. 3D geometry: RDKit ETKDG (fixed seed) and MMFF, so the same SMILES always gives the same input.
  2. Hamiltonian: PySCF Hartree-Fock, STO-3G, an active space of at most 4 electrons in 4 orbitals around
     the Fermi level, Jordan-Wigner mapping (8 qubits; H2 has 4).
  3. Ground state: VQE with a UCCSD ansatz started from the Hartree-Fock state, L-BFGS-B, on the exact
     statevector simulator of PennyLane (no sampling noise).
  4. First excited singlet: a quantum-equation-of-motion style subspace expansion around the VQE state.
     The Hamiltonian is diagonalised in the space spanned by the VQE state and the states made from it by
     spin-free single and double excitations; the second singlet root is the excited state. It takes a
     fraction of a second and needs no second optimisation. (VQD, a second optimisation with an overlap
     penalty, was tried first: 227-361 s per molecule, and for H2 it converged to the wrong singlet.)
  5. gap = E(excited) - E(ground) in eV. This is an excitation energy inside a small active space of a
     minimal basis: qualitative only, and not the same thing as the Hartree-Fock HOMO-LUMO gap, which is
     reported beside it as a classical reference.
  6. Both energies are compared with full CI of the same active space (PySCF). The run is converged when
     the larger error is at or below the threshold (1.6 mHa), otherwise it is flagged (FR-027).

PennyLane and PySCF are installed only for the QML worker (requirements-qml.txt, Linux only: PySCF has no
Windows build). This module imports PennyLane at the top, so import it only when that backend is chosen.
"""
# RDKit is a C extension, so pylint cannot see its members.
# pylint: disable=no-member,c-extension-no-member,invalid-name
import time
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import pennylane as qml  # pylint: disable=import-error
import pennylane.numpy as pnp  # pylint: disable=import-error
import pyscf  # pylint: disable=import-error
from pennylane import qchem  # pylint: disable=import-error
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.optimize import minimize

from backend.workers.qml import hamiltonian
from backend.workers.qml.backend import GapEstimate, QuantumBackend, QuantumCancelled
from backend.workers.qml.convergence import ConvergenceRecord
from backend.workers.qml.references import BASIS, HA_TO_EV, active_space

MAPPING = "jordan_wigner"
ANSATZ = "UCCSD (from the Hartree-Fock state)"
OPTIMISER = "L-BFGS-B"
SIMULATOR = "PennyLane default.qubit, exact statevector"
GEOMETRY = "RDKit ETKDG (seed 7) + MMFF"
GEOMETRY_SEED = 7
EXCITED_METHOD = "qEOM-style subspace expansion (spin-free single and double excitations of the VQE state)"
SPIN_PENALTY = 5.0        # keeps triplet leftovers out of the excited-state search
RANK_CUTOFF = 1e-9        # eigenvalues of the overlap matrix below this are linearly dependent states
ENERGY_DECIMALS = 8       # energies in the stored history, in hartree
MHA = 1000.0


class VqeBackend(QuantumBackend):
    """PennyLane VQE / VQD on a statevector simulator."""

    name = "vqe"
    version = "1.0.0"
    is_stub = False

    def __init__(self, active_electrons: int = 4, active_orbitals: int = 4, max_iterations: int = 300):
        self.active_electrons = active_electrons
        self.active_orbitals = active_orbitals
        self.max_iterations = max_iterations

    def identity(self):
        return {**super().identity(), "active_electrons": self.active_electrons,
                "active_orbitals": self.active_orbitals, "max_iterations": self.max_iterations,
                "excited_method": EXCITED_METHOD, "basis": BASIS, "mapping": MAPPING, "ansatz": ANSATZ,
                "optimiser": OPTIMISER, "geometry": GEOMETRY, "spin_penalty": SPIN_PENALTY,
                "rank_cutoff": RANK_CUTOFF, "pennylane": qml.__version__, "pyscf": pyscf.__version__}

    def estimate_gap(self, smiles: str, threshold_mha: float,
                     should_cancel: Optional[Callable[[], bool]] = None) -> GapEstimate:
        if not isinstance(smiles, str) or not smiles.strip():
            raise ValueError("a structure is required")
        symbols, coords, charge = geometry(smiles)
        return self.compute(symbols, coords, threshold_mha, should_cancel, charge)

    def compute(self, symbols: Sequence[str], coords_angstrom: Sequence[Sequence[float]], threshold_mha: float,
                should_cancel: Optional[Callable[[], bool]] = None, charge: int = 0) -> GapEstimate:
        """The calculation for a geometry (tests call this with H2 and LiH, which are not drug-like)."""
        started = time.monotonic()
        electrons, orbitals = self._active_space(symbols, charge)
        space = active_space(symbols, coords_angstrom, electrons, orbitals, charge)
        reference = space.references
        problem = _Problem(space, electrons, orbitals)
        ground = problem.ground_state(self.max_iterations, should_cancel)
        excited = problem.excited_state(ground)
        error_ground = (ground.energy - reference["fci_ground_energy_ha"]) * MHA
        error_excited = (excited.energy - reference["fci_excited_energy_ha"]) * MHA
        error = max(abs(error_ground), abs(error_excited))
        history = ground.history
        change = abs(history[-1] - history[-2]) * MHA if len(history) > 1 else 0.0
        record = ConvergenceRecord(ground.iterations, round(change, 6),
                                   float(threshold_mha), round(error, 6))
        gap_ev = (excited.energy - excited.ground_energy) * HA_TO_EV
        details = {
            "references": {**{k: round(v, ENERGY_DECIMALS) for k, v in reference.items()},
                           "source": f"PySCF {pyscf.__version__}, Hartree-Fock and full CI of the same "
                                     f"{electrons} electrons in {orbitals} orbitals, {BASIS}"},
            "run": {"mapping": MAPPING, "ansatz": ANSATZ, "optimiser": OPTIMISER, "simulator": SIMULATOR,
                    "excited_state_method": EXCITED_METHOD, "basis": BASIS, "qubits": 2 * orbitals, "geometry": GEOMETRY,
                    "library": f"PennyLane {qml.__version__}", "seconds": round(time.monotonic() - started, 1)},
            "ground_state": _ground_block(ground, reference["fci_ground_energy_ha"]),
            "excited_state": _excited_block(excited, reference["fci_excited_energy_ha"]),
            "gap_error_ev": round(gap_ev - reference["fci_gap_ev"], 6),
        }
        return GapEstimate(round(gap_ev, 6), record, electrons, orbitals, details)

    def _active_space(self, symbols: Sequence[str], charge: int) -> Tuple[int, int]:
        """(electrons, orbitals) of the active space, never larger than the molecule has."""
        total = sum(Chem.GetPeriodicTable().GetAtomicNumber(s) for s in symbols) - charge
        if total <= 0 or total % 2:
            raise ValueError("only closed-shell molecules are supported")
        basis_size = sum(_basis_functions(s) for s in symbols)
        orbitals = min(self.active_orbitals, basis_size)
        electrons = min(self.active_electrons, total, 2 * orbitals)
        if electrons % 2 or electrons == 0:
            raise ValueError("the active space needs an even, non-zero number of electrons")
        return electrons, orbitals


def _basis_functions(symbol: str) -> int:
    """Number of STO-3G basis functions of an atom."""
    return 1 if symbol in ("H", "He") else 5


def geometry(smiles: str) -> Tuple[List[str], List[List[float]], int]:
    """Atoms, Angstrom coordinates and total charge for a SMILES (hydrogens added, MMFF-relaxed)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("the structure could not be read")
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=GEOMETRY_SEED) != 0:
        raise ValueError("no 3D geometry could be built for the structure")
    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    conf = mol.GetConformer()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    coords = [[float(v) for v in conf.GetAtomPosition(i)] for i in range(mol.GetNumAtoms())]
    return symbols, coords, sum(atom.GetFormalCharge() for atom in mol.GetAtoms())


class _Fit:
    """One optimised state: its energy, the optimiser parameters, the energy after each iteration."""

    def __init__(self, energy: float, params: np.ndarray, history: List[float], iterations: int, state):
        self.energy = energy
        self.params = params
        self.history = history
        self.iterations = iterations
        self.state = state


class _Excited(NamedTuple):
    """The excited singlet from the subspace expansion, and the ground energy of the same subspace."""

    energy: float
    ground_energy: float
    dimension: int


def _ground_block(fit: _Fit, fci_energy: float) -> Dict[str, object]:
    return {"energy_ha": round(fit.energy, ENERGY_DECIMALS),
            "error_vs_fci_mha": round((fit.energy - fci_energy) * MHA, 6),
            "iterations": fit.iterations,
            "iteration_energies_ha": [round(e, ENERGY_DECIMALS) for e in fit.history],
            "iteration_errors_vs_fci_mha": [round((e - fci_energy) * MHA, 6) for e in fit.history]}


def _excited_block(excited: _Excited, fci_energy: float) -> Dict[str, object]:
    return {"energy_ha": round(excited.energy, ENERGY_DECIMALS),
            "error_vs_fci_mha": round((excited.energy - fci_energy) * MHA, 6),
            "subspace_ground_energy_ha": round(excited.ground_energy, ENERGY_DECIMALS),
            "subspace_dimension": excited.dimension}


class _Problem:
    """The Hamiltonian of one molecule in its active space and the ansatz that is optimised on it."""

    def __init__(self, space, electrons: int, orbitals: int):
        self.electrons, self.orbitals, self.qubits = electrons, orbitals, 2 * orbitals
        wires = list(range(self.qubits))
        self.h = pnp.array(hamiltonian.hamiltonian(space.core_energy, space.one_body, space.two_body))
        self.s2 = pnp.array(hamiltonian.spin_squared(orbitals))
        qubits = self.qubits
        singles, doubles = qchem.excitations(electrons, qubits)
        s_wires, d_wires = qchem.excitations_to_wires(singles, doubles)
        self.parameters = len(singles) + len(doubles)
        start = qchem.hf_state(electrons, qubits)
        device = qml.device("default.qubit", wires=qubits)

        @qml.qnode(device, diff_method="backprop")
        def circuit(params):
            qml.UCCSD(params, wires=wires, s_wires=s_wires, d_wires=d_wires, init_state=start)
            return qml.state()

        self.state = circuit

    def _expectation(self, matrix, psi):
        return pnp.real(pnp.sum(pnp.conj(psi) * (matrix @ psi)))

    def energy(self, params):
        """<psi|H|psi> of the ansatz state."""
        return self._expectation(self.h, self.state(params))

    def _minimise(self, cost, x0, max_iterations: int, should_cancel) -> Tuple[object, List[float]]:
        """L-BFGS-B with exact gradients; checks for cancellation at every evaluation.

        Returns the scipy result and the true energy <H> after each iteration (for the excited state this is
        not the penalised cost that is minimised)."""
        history: List[float] = []

        def fun(x):
            if should_cancel is not None and should_cancel():
                raise QuantumCancelled("cancelled")
            params = pnp.array(x, requires_grad=True)
            return float(cost(params)), np.array(qml.grad(cost)(params), dtype=float)

        def step(xk):
            history.append(float(self.energy(pnp.array(xk))))

        result = minimize(fun, x0, jac=True, method="L-BFGS-B", callback=step,
                          options={"maxiter": max_iterations})
        if not history:
            history.append(float(self.energy(pnp.array(result.x))))
        return result, history

    def ground_state(self, max_iterations: int, should_cancel) -> _Fit:
        """VQE from the Hartree-Fock state (all parameters zero)."""
        result, history = self._minimise(self.energy, np.zeros(self.parameters), max_iterations, should_cancel)
        psi = np.array(self.state(pnp.array(result.x)))
        return _Fit(float(result.fun), result.x, history, int(result.nit), psi)

    def excited_state(self, ground: _Fit) -> "_Excited":
        """Lowest excited singlet from a subspace expansion around the VQE state.

        The states are O|psi0> for O = 1 and every product of up to two spin-free excitation operators
        E_pq = sum over spin of a+_p a_q. They are singlets when psi0 is. The Hamiltonian (plus a small
        total-spin term) is diagonalised in the orthonormalised span; roots are sorted, and the second
        root is the excited singlet and the first is the refined ground state."""
        h, s2 = np.array(self.h), np.array(self.s2)
        psi0 = np.asarray(ground.state, dtype=complex)
        one_body = [op for row in hamiltonian.excitation_operators(self.orbitals) for op in row]
        singles = [op @ psi0 for op in one_body]
        doubles = [a @ b for a in one_body for b in singles]
        vectors = np.array([psi0, *singles, *doubles]).T                    # (dimension, states)
        overlap = vectors.conj().T @ vectors
        values, rotation = np.linalg.eigh(overlap)
        keep = values > RANK_CUTOFF
        basis = vectors @ (rotation[:, keep] / np.sqrt(values[keep]))       # orthonormal columns
        matrix = basis.conj().T @ (h + SPIN_PENALTY * s2) @ basis
        roots = np.linalg.eigvalsh((matrix + matrix.conj().T) / 2)
        if len(roots) < 2:
            raise RuntimeError("the excitation space has no second state")
        return _Excited(float(roots[1]), float(roots[0]), int(keep.sum()))
