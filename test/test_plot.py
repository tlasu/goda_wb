from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR / "src"))

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt

import goda_wb.plot as gplot


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# chart_to_pixel() / chart_index()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("column", ["H1_3", "H1_250"])
@pytest.mark.parametrize("k", range(4))
def test_chart_to_pixel_maps_frame_corners(column: str, k: int) -> None:
    """図枠の四隅 (0,0), (4,0), (0,ymax), (4,ymax) が較正した画素座標に一致する。"""
    chart = gplot.GODA1975_CHARTS[column]
    corners, ymax = chart["corners"][k], chart["ymax"]
    px, py = gplot.chart_to_pixel([0, 4, 0, 4], [0, 0, ymax, ymax], corners, ymax)
    expected = np.array([corners[c] for c in ("LB", "RB", "LT", "RT")])
    np.testing.assert_allclose(np.column_stack([px, py]), expected)


def test_chart_to_pixel_accepts_scalars() -> None:
    """スカラーを渡しても長さ 1 の配列が返る。"""
    chart = gplot.GODA1975_CHARTS["H1_3"]
    px, py = gplot.chart_to_pixel(2.0, 1.5, chart["corners"][0], chart["ymax"])
    assert px.shape == (1,) and py.shape == (1,)


@pytest.mark.parametrize(("tant", "k"), [(0.1, 0), (0.05, 1), (1 / 30, 2), (0.01, 3)])
def test_chart_index(tant: float, k: int) -> None:
    assert gplot.chart_index(tant) == k


@pytest.mark.parametrize("tant", [1 / 50, 0.02, 0.2])
def test_chart_index_rejects_unsupported_slope(tant: float) -> None:
    with pytest.raises(ValueError):
        gplot.chart_index(tant)


# ---------------------------------------------------------------------------
# 同梱画像
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "image",
    [c["image"] for c in gplot.GODA1975_CHARTS.values()]
    + [f["image"] for f in gplot.GODA1975_APPROX_FIGS.values()],
)
def test_bundled_images_contain_calibrated_frames(image: str) -> None:
    """同梱画像が読み込め、較正した図枠の四隅が画像内にある。"""
    img = gplot.load_chart_image(image)
    height, width = img.shape[:2]
    corner_sets = [
        c
        for chart in gplot.GODA1975_CHARTS.values()
        if chart["image"] == image
        for c in chart["corners"]
    ]
    corner_sets += [
        f["corners"] for f in gplot.GODA1975_APPROX_FIGS.values() if f["image"] == image
    ]
    assert corner_sets
    for corners in corner_sets:
        for x, y in corners.values():
            assert 0 <= x < width and 0 <= y < height


# ---------------------------------------------------------------------------
# 描画関数
# ---------------------------------------------------------------------------


def test_draw_goda1975_chart_shows_image() -> None:
    ax = gplot.draw_goda1975_chart(0.01, "Hmax")
    assert len(ax.images) == 1
    assert "Fig. 57" in ax.get_title()


def test_plot_on_goda1975_draws_curves_and_points() -> None:
    """h0l0 ごとに実線と破線、points に点を描く。"""
    ax = gplot.plot_on_goda1975(0.01, [0.01, 0.02], "H1_3", points=[(2.9, 1.02)])
    assert len(ax.images) == 1
    lines = ax.get_lines()
    assert [ln.get_linestyle() for ln in lines] == ["-", "--", "-", "--", "None"]
    assert lines[0].get_label() == "H0'/L0=0.01"
    px, py = lines[-1].get_data()
    chart = gplot.GODA1975_CHARTS["H1_3"]
    ex, ey = gplot.chart_to_pixel(2.9, 1.02, chart["corners"][3], chart["ymax"])
    np.testing.assert_allclose([px[0], py[0]], [ex[0], ey[0]])


def test_plot_on_goda1975_reuses_existing_chart() -> None:
    """原図を描いた Axes に重ねると、原図を描き直さない。"""
    ax = gplot.draw_goda1975_chart(0.1)
    gplot.plot_on_goda1975(0.1, 0.02, ax=ax, approx=False)
    assert len(ax.images) == 1
    assert len(ax.get_lines()) == 1


def test_plot_on_goda1975_rejects_invalid_column() -> None:
    with pytest.raises(ValueError):
        gplot.plot_on_goda1975(0.01, 0.02, "H1_10")


def test_plot_goda1975_approx_fig() -> None:
    ax = gplot.plot_goda1975_approx_fig(61)
    assert len(ax.images) == 1
    assert len(ax.get_lines()) == 4


def test_plot_goda1975_approx_fig_rejects_unknown_figure() -> None:
    with pytest.raises(ValueError):
        gplot.plot_goda1975_approx_fig(60)
