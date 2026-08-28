"""
Correctness tests for features.morgan_features (the ECFP4 matrix loader).

Needs rdkit, which the torch-free CI env does not install, so these importorskip.

The contract worth pinning is that morgan_features is a drop-in alternative to
align_features -- same ``(X, found)`` shape and masking semantics -- because the
experiment registry treats "rdkit" and "morgan" as interchangeable feature choices
at the same call site.
"""
import numpy as np
import pytest

pytest.importorskip("rdkit")

from features import MORGAN_N_BITS, morgan_features  # noqa: E402


def test_shape_dtype_and_binary_values():
    X, found = morgan_features(["CCO", "c1ccccc1"])
    assert X.shape == (2, MORGAN_N_BITS)
    assert X.dtype == np.uint8, "uint8 keeps the full 828k x 2048 matrix at 1.7GB not 6.8GB"
    assert set(np.unique(X)) <= {0, 1}
    assert found.all()


def test_unparseable_smiles_is_masked_not_raised():
    """Bad SMILES must be reported via `found` (like align_features), never raise --
    callers filter X and labels by the mask."""
    X, found = morgan_features(["CCO", "not_a_smiles", "c1ccccc1"])
    assert list(found) == [True, False, True]
    assert X[1].sum() == 0, "unparseable row must be left as zeros"


def test_duplicate_smiles_map_to_identical_rows():
    """Fingerprints are deduped and scattered back by index; a mis-built inverse mapping
    would silently give a compound the wrong features. In long format the same compound
    recurs across assays, so this path is always exercised."""
    X, found = morgan_features(["CCO", "c1ccccc1", "CCO"])
    assert found.all()
    np.testing.assert_array_equal(X[0], X[2])
    assert not np.array_equal(X[0], X[1])


def test_row_order_is_preserved():
    """np.unique sorts, so the inverse mapping must restore the caller's order."""
    smi = ["c1ccccc1", "CCO", "CCC"]          # deliberately not sorted
    X, _ = morgan_features(smi)
    for i, s in enumerate(smi):
        Xi, _ = morgan_features([s])
        np.testing.assert_array_equal(X[i], Xi[0])


def test_matches_sali_bitvector_fingerprints():
    """Model features and SALI similarity must use the SAME fingerprint, or activity-cliff
    negatives would be mined in a different space than the model sees."""
    from sali import morgan_fps

    smi = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]
    X, _ = morgan_features(smi)
    for i, fp in enumerate(morgan_fps(smi)):
        assert set(np.flatnonzero(X[i])) == set(fp.GetOnBits())


def test_radius_and_nbits_are_configurable():
    X, _ = morgan_features(["CC(=O)Oc1ccccc1C(=O)O"], radius=3, n_bits=1024)
    assert X.shape == (1, 1024)
