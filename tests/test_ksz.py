import pytest

import numpy as np

import py21cmfast as p21c


@pytest.fixture(scope="session")
def lc_ksz(perturb_field, max_redshift):
    return p21c.run_lightcone(
        perturb=perturb_field,
        max_redshift=max_redshift,
        lightcone_quantities=(
            "density",
            "velocity",
            "xH_box",
        ),
    )


def test_ksz(lc_ksz):
    ksz = p21c.run_kSZ(lc_ksz)
    assert not any(np.isnan(np.array(ksz.l_s))) and len(ksz.l_s) > 0
    assert np.shape(ksz.kSZ_box) == (lc_ksz.HII_DIM, lc_ksz.HII_DIM)
