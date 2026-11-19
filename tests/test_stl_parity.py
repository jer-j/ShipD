"""shipd_jax.stl_export.gen_stl_trivec against the reference
(Cython) gen_stl on the sample hulls: same triangulation, same vertices."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("stl")

import matplotlib

from shipd_jax import SAMPLE_HULLS
from shipd_jax.reference.hull_parameterization import (
    Hull_Parameterization as HP,
)
from shipd_jax.stl_export import gen_stl_trivec

matplotlib.use("Agg")

HULLS = np.loadtxt(SAMPLE_HULLS, delimiter=",", dtype=np.float64)
# csv[8]'s stern-bulb fillet system is singular: the reference raises
# LinAlgError (see docs/shipd.md, "Faithfully reproduced quirks")
REFERENCE_FAILS = {8}
KW = {"NUM_WL": 40, "PointsPerWL": 200, "bit_AddTransom": 1, "bit_AddDeckLid": 1}


@pytest.mark.parametrize(
    "i", [i for i in range(len(HULLS)) if i not in REFERENCE_FAILS]
)
def test_stl_matches_reference(i, tmp_path):
    v_ref = np.asarray(
        HP(HULLS[i]).gen_stl(**KW, namepath=str(tmp_path / "ref")).vectors
    )
    v_jax = np.asarray(gen_stl_trivec(HULLS[i], **KW))
    assert v_jax.shape == v_ref.shape
    # the reference stores float32 vectors
    np.testing.assert_array_equal(v_jax.astype(np.float32), v_ref)


def test_reference_failure_is_still_a_failure(tmp_path):
    with pytest.raises(np.linalg.LinAlgError):
        HP(HULLS[8]).gen_stl(**KW, namepath=str(tmp_path / "ref"))
