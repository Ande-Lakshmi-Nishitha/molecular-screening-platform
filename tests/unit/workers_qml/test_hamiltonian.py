"""Unit tests for the dense Jordan-Wigner Hamiltonian builder (MSP-28; FR-023).

Needs only NumPy, so it runs in CI without PennyLane or PySCF. The Hubbard dimer has a closed-form
ground-state energy, which checks the whole construction (operators, signs, the two-electron term).

Run from the repo root with:  python -m unittest discover -s tests -t .
"""
# pylint: disable=missing-function-docstring,missing-class-docstring
import unittest

import numpy as np

from backend.workers.qml import hamiltonian as ham


def hubbard_dimer(t, u):
    """Two sites as two orbitals: hopping -t, on-site repulsion U (only (pp|pp) is non-zero)."""
    one = np.array([[0.0, -t], [-t, 0.0]])
    two = np.zeros((2, 2, 2, 2))
    two[0, 0, 0, 0] = two[1, 1, 1, 1] = u
    return one, two


class OperatorTests(unittest.TestCase):
    def test_the_annihilation_operators_obey_the_fermion_rules(self):
        a = ham.annihilation_operators(4)
        identity = np.eye(16)
        for i, ai in enumerate(a):
            for j, aj in enumerate(a):
                with self.subTest(i=i, j=j):
                    np.testing.assert_allclose(ai @ aj.T + aj.T @ ai, identity if i == j else 0 * identity, atol=1e-12)
                    np.testing.assert_allclose(ai @ aj + aj @ ai, 0 * identity, atol=1e-12)

    def test_the_excitation_operators_are_adjoint_pairs(self):
        e = ham.excitation_operators(3)
        for p in range(3):
            for q in range(3):
                np.testing.assert_allclose(e[p][q].T, e[q][p], atol=1e-12)

    def test_particle_number_counts_the_occupied_spin_orbitals(self):
        number = ham.particle_number(2)
        self.assertEqual(sorted(set(np.round(np.diag(number)).astype(int))), [0, 1, 2, 3, 4])
        self.assertEqual(int(np.diag(number)[0b1100]), 2)

    def test_spin_squared_is_zero_for_a_closed_shell_and_two_for_parallel_spins(self):
        s2 = ham.spin_squared(2)
        closed = np.zeros(16)
        closed[0b1100] = 1.0                      # orbital 0 holds an up and a down electron
        parallel = np.zeros(16)
        parallel[0b1010] = 1.0                    # up electron in orbital 0 and in orbital 1
        self.assertAlmostEqual(float(closed @ s2 @ closed), 0.0, places=12)
        self.assertAlmostEqual(float(parallel @ s2 @ parallel), 2.0, places=12)


class HamiltonianTests(unittest.TestCase):
    def test_the_hubbard_dimer_ground_state_matches_the_closed_form(self):
        for t, u in ((1.0, 0.0), (1.0, 4.0), (0.5, 3.0)):
            with self.subTest(t=t, u=u):
                one, two = hubbard_dimer(t, u)
                h = ham.hamiltonian(0.0, one, two)
                number, s2 = ham.particle_number(2), ham.spin_squared(2)
                shifted = h + 50.0 * (number - 2 * np.eye(16)) @ (number - 2 * np.eye(16)) + 50.0 * s2
                expected = (u - np.sqrt(u * u + 16 * t * t)) / 2
                self.assertAlmostEqual(float(np.linalg.eigvalsh(shifted)[0]), expected, places=9)

    def test_the_core_energy_shifts_every_level(self):
        one, two = hubbard_dimer(1.0, 2.0)
        low = np.linalg.eigvalsh(ham.hamiltonian(0.0, one, two))
        high = np.linalg.eigvalsh(ham.hamiltonian(1.5, one, two))
        np.testing.assert_allclose(high - low, 1.5, atol=1e-10)

    def test_it_is_symmetric_and_keeps_the_electron_number(self):
        one, two = hubbard_dimer(0.7, 2.5)
        h, number = ham.hamiltonian(0.3, one, two), ham.particle_number(2)
        np.testing.assert_allclose(h, h.T, atol=1e-12)
        np.testing.assert_allclose(h @ number - number @ h, 0.0, atol=1e-10)

    def test_it_commutes_with_total_spin_for_a_spin_free_hamiltonian(self):
        one, two = hubbard_dimer(0.7, 2.5)
        h, s2 = ham.hamiltonian(0.0, one, two), ham.spin_squared(2)
        np.testing.assert_allclose(h @ s2 - s2 @ h, 0.0, atol=1e-10)

    def test_integrals_of_the_wrong_shape_are_refused(self):
        with self.assertRaises(ValueError):
            ham.hamiltonian(0.0, np.zeros((2, 2)), np.zeros((3, 3, 3, 3)))
        with self.assertRaises(ValueError):
            ham.hamiltonian(0.0, np.zeros((2, 3)), np.zeros((2, 2, 2, 2)))


if __name__ == "__main__":
    unittest.main()
