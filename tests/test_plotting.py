"""
Testing plots is kind of hard, but we just check that it runs through without crashing.
"""

import pytest

from py21cmfast import initial_conditions
from py21cmfast import plotting
from py21cmfast import run_lightcone


def test_coeval_sliceplot():
    ic = initial_conditions(user_params={"HII_DIM": 35, "DIM": 70})

    fig, ax = plotting.coeval_sliceplot(ic)

    assert ax.xaxis.get_label().get_text() == "x-axis [Mpc]"
    assert ax.yaxis.get_label().get_text() == "y-axis [Mpc]"

    with pytest.raises(ValueError):  # bad slice axis
        plotting.coeval_sliceplot(ic, slice_axis=-2)

    with pytest.raises(IndexError):  # tring to plot slice that doesn't exist
        plotting.coeval_sliceplot(ic, slice_index=50)

    fig2, ax2 = plotting.coeval_sliceplot(ic, fig=fig, ax=ax)

    assert fig2 is fig
    assert ax2 is ax

    fig2, ax2 = plotting.coeval_sliceplot(ic, fig=fig)

    assert fig2 is fig
    assert ax2 is ax

    fig2, ax2 = plotting.coeval_sliceplot(ic, ax=ax)

    assert fig2 is fig
    assert ax2 is ax

    fig, ax = plotting.coeval_sliceplot(
        ic, kind="hires_density", slice_index=50, slice_axis=1
    )
    assert ax.xaxis.get_label().get_text() == "x-axis [Mpc]"
    assert ax.yaxis.get_label().get_text() == "z-axis [Mpc]"


@pytest.fixture("module")
def lc():
    return run_lightcone(
        redshift=25, max_redshift=30, user_params={"HII_DIM": 35, "DIM": 70}
    )


def test_lightcone_sliceplot_default(lc):

    fig, ax = plotting.lightcone_sliceplot(lc)

    assert ax.yaxis.get_label().get_text() == "y-axis [Mpc]"
    assert ax.xaxis.get_label().get_text() == "Redshift"


def test_lightcone_sliceplot_vertical(lc):
    fig, ax = plotting.lightcone_sliceplot(lc, vertical=True)

    assert ax.yaxis.get_label().get_text() == "Redshift"
    assert ax.xaxis.get_label().get_text() == "y-axis [Mpc]"


def test_lc_sliceplot_freq(lc):
    fig, ax = plotting.lightcone_sliceplot(lc, zticks="frequency")

    assert ax.yaxis.get_label().get_text() == "y-axis [Mpc]"
    assert ax.xaxis.get_label().get_text() == "Frequency [MHz]"


def test_lc_sliceplot_cdist(lc):
    fig, ax = plotting.lightcone_sliceplot(lc, zticks="comoving_distance")

    assert ax.yaxis.get_label().get_text() == "y-axis [Mpc]"
    assert ax.xaxis.get_label().get_text() == "Comoving Distance [Mpc]"

    xlim = ax.get_xlim()
    assert xlim[0] >= lc.lightcone_distances.min()
    assert xlim[1] <= lc.lightcone_distances.max()


def test_lc_sliceplot_sliceax(lc):
    fig, ax = plotting.lightcone_sliceplot(lc, slice_axis=2)

    assert ax.yaxis.get_label().get_text() == "y-axis [Mpc]"
    assert ax.xaxis.get_label().get_text() == "x-axis [Mpc]"
