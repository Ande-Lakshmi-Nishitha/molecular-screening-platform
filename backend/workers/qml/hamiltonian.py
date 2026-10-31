"""Second-quantised electronic Hamiltonian as a dense matrix, Jordan-Wigner (MSP-28; FR-023).

Owner: Nishitha. Built here from the active-space integrals instead of with a chemistry library, because the
PennyLane routine ignores the active space when the active space holds every electron of the molecule
(LiH, with 4 electrons in 4 of 6 orbitals, came back with 12 qubits). With a dense matrix of 2**8 = 256
rows the construction takes well under a second.

Spin orbitals are interleaved: qubit 2p is orbital p with spin up, qubit 2p + 1 is orbital p with spin down.
A qubit in state |1> is an occupied spin orbital, and qubit 0 is the leftmost factor of every Kronecker
product. This is the ordering used by PennyLane's Hartree-Fock state and UCCSD ansatz.

  H = E_core + sum_pq h[p,q] E_pq + 1/2 sum_pqrs (pq|rs) (E_pq E_rs - delta_qr E_ps)
  E_pq = a+_p(up) a_q(up) + a+_p(down) a_q(down)       (spin-free excitation operator)

Pure NumPy: no RDKit, PennyLane, PySCF, Celery or database imports.
"""
from functools import lru_cache
from typing import Tuple

import numpy as np

_IDENTITY = np.eye(2)
_PARITY = np.diag([1.0, -1.0])
_LOWER = np.array([[0.0, 1.0], [0.0, 0.0]])   # |0><1|: removes an electron from the qubit


@lru_cache(maxsize=None)
def annihilation_operators(qubits: int) -> Tuple[np.ndarray, ...]:
    """Matrices of a_j for j = 0 .. qubits - 1 (a parity string of Z on every qubit before j)."""
    operators = []
    for j in range(qubits):
        matrix = np.eye(1)
        for factor in [_PARITY] * j + [_LOWER] + [_IDENTITY] * (qubits - j - 1):
            matrix = np.kron(matrix, factor)
        operators.append(matrix)
    return tuple(operators)


@lru_cache(maxsize=None)
def excitation_operators(orbitals: int) -> Tuple[Tuple[np.ndarray, ...], ...]:
    """E[p][q] for spatial orbitals p, q (2 * orbitals qubits)."""
    annihilate = annihilation_operators(2 * orbitals)
    return tuple(tuple(sum(annihilate[2 * p + spin].T @ annihilate[2 * q + spin] for spin in (0, 1))
                       for q in range(orbitals)) for p in range(orbitals))


def hamiltonian(core_energy: float, one_body: np.ndarray, two_body: np.ndarray) -> np.ndarray:
    """Dense Hamiltonian from the core energy, h[p,q] and the electron-repulsion integrals (pq|rs)."""
    orbitals = one_body.shape[0]
    if one_body.shape != (orbitals, orbitals) or two_body.shape != (orbitals,) * 4:
        raise ValueError("the integrals do not fit one active space")
    e = excitation_operators(orbitals)
    matrix = core_energy * np.eye(4 ** orbitals)
    for p in range(orbitals):
        for q in range(orbitals):
            matrix = matrix + one_body[p, q] * e[p][q]
            pair = sum(two_body[p, q, r, s] * e[r][s] for r in range(orbitals) for s in range(orbitals))
            matrix = matrix + 0.5 * (e[p][q] @ pair)
            for s in range(orbitals):
                matrix = matrix - 0.5 * two_body[p, q, q, s] * e[p][s]
    return (matrix + matrix.T) / 2


def spin_squared(orbitals: int) -> np.ndarray:
    """S^2 = S- S+ + Sz (Sz + 1)."""
    annihilate = annihilation_operators(2 * orbitals)
    raising = sum(annihilate[2 * p].T @ annihilate[2 * p + 1] for p in range(orbitals))
    sz = 0.5 * sum(annihilate[2 * p].T @ annihilate[2 * p] - annihilate[2 * p + 1].T @ annihilate[2 * p + 1]
                   for p in range(orbitals))
    return raising.T @ raising + sz @ sz + sz


def particle_number(orbitals: int) -> np.ndarray:
    """Total number of electrons."""
    annihilate = annihilation_operators(2 * orbitals)
    return sum(a.T @ a for a in annihilate)
