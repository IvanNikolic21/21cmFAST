import pytest

import numpy as np

import py21cmfast as p21c


def test_ksz(lc):
    ksz = p21c.run_kSZ(lc)
    assert not any(np.isnan(np.array(ksz.l_s))) and len(ksz.l_s) > 0
    assert np.shape(ksz.kSZ_box) == (lc.HII_DIM, lc.HII_DIM)
