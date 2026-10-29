"""Unit tests for MOL/SDF reading (MSP-20; FR-001, FR-004, BR-001).

Run from the repo root with:  python -m unittest discover -s tests -t .
"""
# RDKit is a C extension, so pylint cannot see its members.
# pylint: disable=no-member,c-extension-no-member,missing-function-docstring,missing-class-docstring
import unittest

from rdkit import Chem
from rdkit.Chem import AllChem

from backend.services.molecule.errors import (
    CODE_TOO_LARGE,
    InputTooLargeError,
    http_status_for,
)
from backend.services.molecule.molfile import parse_molblock, read_first_record
from backend.services.molecule.standardise import (
    ERROR_CODE,
    MoleculeError,
    standardise_mol,
    standardise_smiles,
)

BIG = 100_000


def molblock_of(smiles: str) -> str:
    """A MOL block for a SMILES string, with 2D coordinates (like a real file)."""
    mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)
    return Chem.MolToMolBlock(mol)


def mol_file(smiles: str) -> bytes:
    return molblock_of(smiles).encode("utf-8")


def sdf_file(*smiles_list: str) -> bytes:
    return "".join(molblock_of(s) + "$$$$\n" for s in smiles_list).encode("utf-8")


class ReadFirstRecordTests(unittest.TestCase):
    def test_fr001_plain_mol_file_is_one_record(self):
        block, count = read_first_record(mol_file("CCO"), BIG)
        self.assertEqual(count, 1)
        self.assertIn("V2000", block)

    def test_fr001_sdf_uses_only_the_first_record(self):
        block, count = read_first_record(sdf_file("CCO", "CC(=O)N"), BIG)
        self.assertEqual(count, 2)
        self.assertEqual(standardise_mol(parse_molblock(block)).canonical_smiles, "CCO")

    def test_fr001_windows_line_endings_are_accepted(self):
        crlf = mol_file("CCO").replace(b"\n", b"\r\n")
        block, count = read_first_record(crlf, BIG)
        self.assertEqual(count, 1)
        self.assertEqual(standardise_mol(parse_molblock(block)).canonical_smiles, "CCO")

    def test_fr001_blank_title_line_is_kept(self):
        # A MOL block's first line (the title) may be empty; it must not be dropped.
        block, _ = read_first_record(mol_file("CCO"), BIG)
        self.assertEqual(standardise_mol(parse_molblock(block)).canonical_smiles, "CCO")

    def test_fr004_empty_file_is_rejected(self):
        for data in (b"", b"   \n\n  ", b"$$$$\n"):
            with self.subTest(data=data):
                with self.assertRaises(MoleculeError) as ctx:
                    read_first_record(data, BIG)
                self.assertEqual(ctx.exception.code, ERROR_CODE)

    def test_fr004_binary_file_is_rejected(self):
        for data in (b"\x00\x01\x02binary", b"\xff\xfe\xfd"):
            with self.subTest(data=data):
                with self.assertRaises(MoleculeError):
                    read_first_record(data, BIG)

    def test_fr004_non_bytes_input_is_rejected(self):
        with self.assertRaises(MoleculeError):
            read_first_record("CCO", BIG)  # type: ignore[arg-type]

    def test_fr053_file_over_the_limit_is_too_large(self):
        data = mol_file("CCO")
        with self.assertRaises(InputTooLargeError) as ctx:
            read_first_record(data, len(data) - 1)
        self.assertEqual(ctx.exception.code, CODE_TOO_LARGE)
        self.assertEqual(http_status_for(ctx.exception), 413)

    def test_fr053_file_exactly_at_the_limit_is_accepted(self):
        data = mol_file("CCO")
        _, count = read_first_record(data, len(data))
        self.assertEqual(count, 1)


class ParseMolblockTests(unittest.TestCase):
    def test_fr001_ethanol_file_matches_the_smiles_result(self):
        from_file = standardise_mol(parse_molblock(molblock_of("CCO")))
        from_string = standardise_smiles("CCO")
        self.assertEqual(from_file.canonical_smiles, from_string.canonical_smiles)
        self.assertEqual(from_file.inchikey, from_string.inchikey)

    def test_fr001_acetamide_file_matches_the_smiles_result(self):
        from_file = standardise_mol(parse_molblock(molblock_of("CC(=O)N")))
        from_string = standardise_smiles("CC(=O)N")
        self.assertEqual(from_file.canonical_smiles, from_string.canonical_smiles)
        self.assertEqual(from_file.inchikey, from_string.inchikey)

    def test_fr004_garbage_text_is_rejected(self):
        with self.assertRaises(MoleculeError) as ctx:
            parse_molblock("this is not a MOL file\n")
        self.assertEqual(ctx.exception.code, ERROR_CODE)

    def test_fr004_invalid_valence_is_rejected(self):
        # pentavalent carbon: parses, but fails sanitisation
        mol = Chem.MolFromSmiles("C(C)(C)(C)(C)C", sanitize=False)
        block = Chem.MolToMolBlock(mol, kekulize=False)
        with self.assertRaises(MoleculeError) as ctx:
            parse_molblock(block)
        self.assertIn("sanitisation", ctx.exception.message)

    def test_br001_file_with_no_atoms_is_rejected(self):
        empty = Chem.MolToMolBlock(Chem.Mol())
        with self.assertRaises(MoleculeError):
            standardise_mol(parse_molblock(empty))


if __name__ == "__main__":
    unittest.main()
