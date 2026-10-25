# CLAUDE.md — AI Assistant Guide for `goda-wb`

This file provides context for AI assistants (Claude Code and others) working on this repository.

---

## Project Overview

**goda-wb** is a Python library implementing Goda's breaking wave deformation model for coastal engineering. It computes wave transformation in the surf zone, accounting for wave breaking, shoaling, and wave setup using probabilistic wave height distributions.

- **Primary reference**: Goda (1975) — irregular wave breaking and surf zone hydrodynamics
- **Author**: tlasu
- **License**: MIT
- **Version**: 0.1.0

---

## Repository Structure

```
goda_wb/
├── src/
│   └── goda_wb/
│       ├── __init__.py       # Package init — imports the `core` submodule only
│       ├── core.py           # Goda (1975) model — all its computation logic (744 lines, 18 functions)
│       ├── pegbis.py         # Goda (2004) gradational breaking model PEGBIS, 1-D cross-shore (451 lines)
│       └── constant.py       # Physical constants & model parameters (191 lines)
├── test/
│   ├── test_core.py          # Test suite for core (662 lines, 137 test cases)
│   └── test_pegbis.py        # Test suite for pegbis (366 lines, 61 test cases)
├── sample/                   # Usage examples and visualization scripts
│   ├── check_core.py
│   ├── check_goda.py
│   ├── check_shuto.py
│   ├── compare_goda1975.py   # Overlay results on Goda (1975) original charts
│   ├── goda1975/             # Original chart scans (p.96–97, Figs. 61–62) and comparison output
│   ├── compare_pegbis.py     # PEGBIS vs Goda (1975) on uniform slopes, and Kweon's step (Fig. 11)
│   ├── compare_pegbis_paper.py # Overlay PEGBIS on the paper's 付図-4〜7 and Fig. 11 (page images not committed)
│   ├── pegbis/               # compare_pegbis.py / compare_pegbis_paper.py output
│   ├── draw_on_image.py
│   ├── draw_on_image2.py
│   └── run.py
├── .github/
│   └── workflows/
│       └── ci.yml            # GitHub Actions: lint + multi-version test matrix
├── pyproject.toml            # Build config, dependencies, ruff/pytest settings
├── uv.lock                   # Locked dependency versions
├── .python-version           # Python 3.11
├── skill.md                  # Technical reference (Japanese + English)
└── README.md                 # User documentation (Japanese)
```

---

## Technology Stack

| Category          | Tool/Library          | Version     |
|-------------------|-----------------------|-------------|
| Language          | Python                | >= 3.11     |
| Package manager   | uv                    | latest      |
| Build backend     | uv_build              | 0.8.8–0.9.0 |
| Core numeric      | numpy                 | >= 2.3.4    |
| Data output       | pandas                | >= 2.3.3    |
| Testing           | pytest                | >= 9.0.0    |
| Linting/format    | ruff                  | >= 0.14.9   |
| Validation (dev)  | scipy                 | >= 1.16.3   |
| Visualization     | matplotlib            | >= 3.10.8   |
| Notebooks         | ipykernel             | >= 7.1.0    |

---

## Development Workflow

### Setup

```bash
# Install dependencies (runtime + dev)
uv sync --group dev

# Or just runtime dependencies
uv sync
```

### Running Tests

```bash
uv run pytest -v
```

Tests live in `test/test_core.py` (137 cases, covering all public API functions of `core`, including a comparison with values read from Goda (1975) Figs. 50–57 and Table-4) and `test/test_pegbis.py` (61 cases for `pegbis`, including values digitized from the paper's 付図-4〜7, tolerance 0.045). All tests must pass before committing.

### Linting and Formatting

```bash
# Check linting
uv run ruff check src/ test/

# Check formatting (no auto-fix)
uv run ruff format --check src/ test/

# Apply fixes
uv run ruff check --fix src/ test/
uv run ruff format src/ test/
```

**Ruff configuration** (in `pyproject.toml`):
- Line length: **100 characters**
- Target Python: **3.11**
- Active rule sets: `E`, `F`, `W`, `I` (isort), `UP` (pyupgrade), `B` (bugbear), `SIM` (simplify)
- Per-file ignore: `E402` in `test/` (import order)

### CI/CD

GitHub Actions runs on every push/PR to `main`:
1. **Lint job**: ruff check + format check
2. **Test job**: pytest on Python 3.11, 3.12, 3.13 (matrix strategy)

---

## Public API

All functions are in `goda_wb.core`. They are **not** re-exported from the top-level package, so use `import goda_wb.core as core` (or `goda_wb.core.<func>`).

### Primary Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `cal_surf_goda` | `(tant, h0l0, dl0=None) → DataFrame` | Main dimensionless wave transformation calculation |
| `cal_surf_goda_dim` | `(tant, H0, T, dim=False, d=None) → DataFrame` | Dimensional wrapper (SI units) |
| `cal_surf_goda_point` | `(tant, h0l0, dl0) → DataFrame` | Single-point calculation (same as `cal_surf_goda(tant, h0l0, [dl0])`) |
| `cal_surf_goda_approx` | `(tant, h0l0, dl0=None) → DataFrame` | Goda (1975) simplified formulas, eqs. (67)/(69); columns `H1_3`, `Hmax`, `dh0`, `dl0`, `aks` (no setup, no probability model) |
| `approx_coef` | `(tant, h0l0) → dict` | Coefficients of eqs. (68)/(70): `beta0`, `beta1`, `betamax`, `beta0_max`, `beta1_max`, `betamax_max` |
| `cal_h0_from_height` | `(tant, H, T, d, col="H1_3", approx=False) → float` | Inverse: deep-water height H0 [m] whose wave height `col` at depth `d` [m] equals `H` [m]; `approx=True` uses the simplified formulas (`col` = `H1_3` or `Hmax`) |
| `shoal` | `(dl0, h0l0) → float` | Shoaling coefficient (3-region model) |
| `aksi` | `(dl0) → float` | Linear (small-amplitude) shoaling coefficient |
| `cal_wave_length` | `(d, T) → float` | Wave length from depth and period |
| `wave` | `(d) → float` | Solve dispersion relation: x·tanh(x) = d |

**Computation grid.** Setup must be marched from offshore to onshore over a fine grid. When `dl0` is given, `cal_surf_goda` merges it with the points of `goda_dl0_list` deeper than `min(dl0)`, marches over that merged grid (sorted descending) via the private `_march_profile()`, and returns only the requested rows, in the caller's order. Consequently `dl0` may be in any order, may contain duplicates (identical rows are returned), and a sparse list such as `[0.1, 0.03, 0.01]` gives the same values as the corresponding rows of the default profile. An empty `dl0` raises `ValueError`.

### Output DataFrame Columns

| Column | Description |
|--------|-------------|
| `H1_1000` through `H1_3` | Representative wave heights (exceedance: 1/1000, 1/250, …, 1/3) |
| `dh0` | Depth-to-deep-water-height ratio |
| `dl0` | Depth-to-deep-water-wavelength ratio |
| `etal0` | Mean water level change ratio (η/L₀); 0 at the most offshore point, negative for set-down, positive for setup |
| `aks` | Shoaling coefficient |
| `d`, `H1_3_dim`, … | Dimensional values (only when `dim=True`) |

### PEGBIS (`goda_wb.pegbis`)

A separate model: Goda (2004) ECOH/YG No.2 Ver.5, restricted to 1-D cross-shore, normal incidence. Import with `import goda_wb.pegbis as pegbis`; it is not re-exported either. It reuses only `core.wave()` and shares no code path with `cal_surf_goda`.

| Function | Signature | Description |
|----------|-----------|-------------|
| `cal_surf_pegbis` | `(x, h, H0, T, fw=0.01, M=831, n_freq=1, cb=None, cb2=None, alpha_b=None) → DataFrame` | Arbitrary profile; `x` [m] increasing shoreward, `h` [m] > 0. Columns `x`, `h`, `slope`, `Ks`, `H1_250` … `H1_3`, `Hrms`, `Hmean` [m]. `fw` scalar or per-point array. `cb=None` → `default_cb(M)`; `cb2=None` → `0.080 · cb / 0.080` (proportional, as the paper does for its field cases; it recommends `cb=0.070` for field sites). `alpha_b=None` → Ver.5 slope function; `0.125` reproduces the paper's main-text figures (Ver.3 and earlier) |
| `cal_surf_pegbis_slope` | `(tant, h0l0, dl0=None, fw=0.01, M=831, n_freq=1, cb=None, cb2=None, alpha_b=None) → DataFrame` | Uniform slope, dimensionless (L0 = 1, start at h/L0 = max(0.5, max(dl0))); columns as `cal_surf_goda` (ratios to H0') plus `Hrms`, `Hmean`; no `etal0` |
| `gradational_breaker_index` | `(h, L0, slope, Hm, H1, cb=0.080, cb2=0.080)` | κm, eq. (7)/(8) with the β0-term slope factor exp(60 I²) (calibrated; the paper prints 30, see note 8); for slope <= 0 eq. (11) constants with I = 0; h/L0 capped at 0.2 |
| `breaking_decay_coef` | `(a, h, kappa, slope, alpha_b=None)` | f_Db, eq. (4) with α_b = 3/8 (0.3 + 2.5 max(I, 0)) (appendix eq. 付1) |
| `bottom_friction_coef` | `(a, h, kh, fw)` | f_Df, eq. (14) |
| `swart_fw` | `(ks, am) → float` | Swart (1974) friction factor, capped at 0.30 |
| `wave_levels` / `spectral_periods` | `(H0, M)` / `(T, n_freq)` | Equal-probability Rayleigh levels, eq. (5), with H1 raised to at least 1.80 H0' for small M (paper §6); equal-energy frequency split, eq. (15) with Tp = 0.6^(-1/4) T1/3 = 1.136 T1/3 (modified Bretschneider–Mitsuyasu; paper appendix: Tp = 9.10 s ↔ T1/3 = 8.01 s) |
| `default_cb` | `(M) → float` | Cb by number of levels: 0.080 (M ≥ 300), 0.077 (101), 0.076 (61), 0.074 (31), linear in between |

---

## Code Conventions

### Naming

- **Parameter notation** follows Japanese coastal engineering conventions:
  - `tant` = tan θ (seabed slope)
  - `h0l0` = H₀/L₀ (deep water wave steepness)
  - `dl0` = d/L₀ (relative depth)
  - `dh0` = d/H₀ (depth to height ratio)
- Internal/helper functions (`etai`, `sbeat`, `bindx`, `setup`, `out`, `prob`, `prob01`) are not prefixed and are interleaved with the public functions in `core.py`
- Trailing underscores (`xp_`, `p_`, `aks_` in `cal_surf_goda`) denote per-point accumulator lists

### Module Organization

- Keep all computation logic of the Goda (1975) model in `src/goda_wb/core.py`, and of PEGBIS in `src/goda_wb/pegbis.py`
- Keep all constants (physical, numerical, model) in `src/goda_wb/constant.py`
- Do not add new modules without clear necessity — the codebase is intentionally compact

### Style

- Line length: 100 characters (enforced by ruff)
- Type hints are used throughout — maintain them in any additions
- Docstrings, comments and error messages are in Japanese (Google style `Args:` / `Returns:` sections)
- No global state, but several helpers (`etai`, `setup`, `prob`, `prob01`) update the lists passed to them in place and also return them

### Constants (from `constant.py`)

| Name | Value | Purpose |
|------|-------|---------|
| `g` | 9.81 | Gravitational acceleration (m/s²) |
| `pi2` | 2π | Convenience constant |
| `MP` | 51 | Discretization points for probability density |
| `NWAVE` | 7 | Number of representative wave heights |
| `CONVERGENCE_TOL` | 1e-6 | Newton method convergence tolerance |
| `WAVE_CONVERGENCE_TOL` | 0.0003 | Wave dispersion iteration tolerance |
| `MAX_ITERATIONS` | 100 | Iteration cap for `wave()` / `shoal()`; exceeding it raises `RuntimeError` |
| `REPRESENTATIVE_WAVE_PROBABILITIES` | 7 values | Exceedance probabilities for H1/1000 … H1/3 |
| `DEEP_WATER_THRESHOLD` | 10.0 | dl0 threshold for deep water classification |
| `APPROX_SHALLOW_DL0` | 0.2 | h/L0 below which the simplified formulas apply breaking (eqs. 67/69) |
| `RAYLEIGH_TAIL_LIMIT` | 4.0 | Upper bound of the wave-height grid in `prob()`, as `a·x` (Rayleigh tail ≈ exp(−16)) |
| `ETAI_MAX_RATIO` | 2.0 | Cap on the spacing ratio used by `etai()` extrapolation |
| `INVERSE_H0L0_MIN` / `INVERSE_H0L0_MAX` | 1e-4 / 0.1 | H0/L0 search bracket of `cal_h0_from_height()`; targets outside it raise `ValueError` |
| `INVERSE_TOL` | 1e-6 | Convergence tolerance on log(H0/L0) in `cal_h0_from_height()` |
| `PEGBIS_*` | — | PEGBIS constants: `CB`/`BETA0`/`P` = 0.080/0.016/0.333 (eq. 8), `CB2`/`BETA02`/`P2` = 0.080/0.016/0.667 (eq. 11; paper gives Cb' = 0.070–0.080), `LEVELS` = 831, `FW` = 0.01, `BREAKING_DL0` = 0.2, `SLOPE_DISTANCE_L0` = 0.1, `CB_BY_LEVELS` (table for `default_cb`), `H1_MIN_RATIO` = 1.80, `KAPPA_SLOPE_COEF` = 60 (calibrated, paper prints 30), `TP_T13` = 1.136, internal grid limits `MAX_DH_RATIO`/`MAX_DX_H_RATIO`/`MAX_DX_L_RATIO` |
| `goda_dl0_list` | 113 values | Default d/L₀ evaluation points (1e-9–1.0, descending) |
| `GAUSS_HERMITE_ABSCISSAE/WEIGHTS` | 8 points | Discretized normal distribution for surf beat (see note below) |

---

## Testing Conventions

- Tests are in `test/test_core.py`, using standard `pytest`
- Use `pytest.mark.parametrize` for boundary/edge case sweeps
- Use numerical regression tests (assert computed values ≈ expected reference values)
- Validate against the original paper: `GODA1975_CHART_H13` / `GODA1975_CHART_HMAX` hold values read from Goda (1975) Figs. 50–57 (tolerance 0.04); `GODA1975_TABLE4` holds Table-4 coefficients for the simplified formulas (tolerance 0.01)
- Test property invariants (e.g., `H1_3 <= H1_1000` always holds)
- Test `ValueError` is raised for invalid inputs
- No mocking — all tests exercise the real numeric implementation

### Running a Single Test

```bash
uv run pytest -v test/test_core.py::test_function_name
```

---

## Key Implementation Notes

1. **Wave dispersion**: `wave(d)` solves x·tanh(x) = d iteratively (Newton's method), converging to `WAVE_CONVERGENCE_TOL`. Returns `d` unchanged above `DEEP_WATER_THRESHOLD` and `0.0` for `d == 0`.

2. **Shoaling model**: `shoal()` (Goda's nonlinear shoaling) splits into 3 regions using thresholds `d30l0` and `d50l0`, each found by Newton iteration:
   - `dl0 >= d30l0`: linear (small-amplitude) theory, `aksi(dl0)`
   - `d50l0 < dl0 < d30l0`: power law `aksi30 * (d30l0 / dl0) ** (2/7)`
   - `dl0 <= d50l0`: shallow-water region, solved for Ks by Newton iteration

3. **Probability distribution**: `prob()` uses a Rayleigh distribution with breaking correction to compute the full wave height probability density across `MP=51` points on `[0, min(x1, RAYLEIGH_TAIL_LIMIT / a)]`, so the grid follows the breaking limit near shore and stays fine offshore where `x1` is huge. The breaking limits come from `bindx()`, which implements Goda (1975) eq. (3) with `K = 15`, `s = 4/3` and eq. (8) `A1 = 0.18`, `A2 = 0.12` (verified against the original paper, p.66).

4. **Surf beat**: `sbeat()` integrates over 8 surf-beat water level offsets. Despite the `GAUSS_HERMITE_*` names, the constants are a discretized standard normal distribution (weights 0.3413, 0.1359, … are band probabilities summing to 1), not Gauss–Hermite nodes/weights.

5. **Setup calculation**: `setup()` updates the mean water level by a single explicit step from the radiation-stress (energy) balance between point `i-1` and `i` — no iteration. `etai()` linearly extrapolates the previous two points to give the initial estimate used for breaking; the spacing ratio is capped at `ETAI_MAX_RATIO` because user points close to default points would otherwise make the extrapolation blow up. This marching requires `dl0` ordered offshore → onshore on a fine grid, which `cal_surf_goda` guarantees (see **Computation grid** above).

6. **Simplified formulas**: `cal_surf_goda_approx()` implements Goda (1975) p.99 eqs. (67)–(70) using `shoal()` for Ks. The paper prints β0* = 0.52 (H0'/L0)^-0.38…, but Table-4 confirms 0.052, which is what the code uses. Table-4 also lists βmax = 1.31 for slope 1/30, H0'/L0 = 0.01 where eq. (68) gives 1.318 (a rounding slip in the table).

7. **Inverse calculation**: `cal_h0_from_height()` fixes T (so d/L0 is fixed) and solves `H(H0) = H` on log(H0/L0) with the Illinois (modified regula falsi) method, using `cal_surf_goda` or `cal_surf_goda_approx` as the forward model; the wave height is monotonically increasing in H0. One forward evaluation of the full model takes ~0.1 s, so one inversion takes about 1 s. Inside the surf zone H saturates with H0, so the inverse is ill-conditioned there.

8. **PEGBIS**: with normal incidence the parabolic equation reduces, per level m and frequency component n, to `da/dx = -(1/2) a d(ln cg)/dx - f_D a`. `cal_surf_pegbis` marches it with Heun's method on an internal grid (input points plus subdivisions limited by `PEGBIS_MAX_*`) from the offshore boundary, where heights are linearly shoaled from deep water (the boundary must be outside the surf zone). κm always uses the L0 of T1/3, also for spectral components. Level heights combine components as rms and are sorted before computing H1/n (fractional level weighting); Hmax is represented by `H1_250` as in the paper. No setup, surf beat, nonlinear shoaling, refraction or diffraction. Calibration: with exp(30 I²) as printed in eq. (7), Hmax on slope 1/10 came out ~0.07 H0' below the paper's own results in both 付図-4 (Ver.5, α_b(I)) and Fig. 6 (Ver.4 figure, α_b = 0.125), and on Kweon's step breaking started too early (Hmax peak at x = −1.35 m vs −1.1 m), while slopes ≤ 1/20 matched. Since the shortfall is independent of α_b, it is in κ; fitting the β0-term factor exp(C I²) to digitized 付図-4〜7 gives C ≈ 55–65, which also fixes Fig. 6 and Fig. 11 and is indistinguishable from 30 in Fig. 2 (I = 1/20). Rejected alternatives: grid lag of an explicit scheme (raises H1/3 too much), L0 from Tp, local (Hm/H1), H1 = 1.80 H0' for M = 831 (worsens gentle slopes). Remaining differences: Kweon's offshore H1/3 is ~3 % above the paper (the paper seems to impose H0' at the boundary without shoaling; Ks ≈ 1.03), and Hrms/Hmean on the step are 6–10 % low. Shoaling minimum 0.937 with `n_freq=10` matches the paper.

---

## Sample Scripts

Located in `sample/` — not part of the installable package, intended for exploration and visualization:

| Script | Purpose |
|--------|---------|
| `check_core.py` | Basic API demonstration |
| `check_goda.py` | Wave transformation plots across steepness values |
| `check_shuto.py` | Shoaling coefficient curves (`shoal()`) for comparison with Shuto's shoaling chart |
| `compare_goda1975.py` | Overlays `H1_3` / `H1_250` (solid) and the simplified formulas (dashed) on Goda (1975) Figs. 50–57 using calibrated chart-frame corners, and reproduces Figs. 61–62 (simplified formulas vs. computed values); writes `goda1975/compare_*.png`. Fig. 52 (slope 1/30) is calibrated on its grid instead, because its bottom frame line and y labels are drawn 0.03–0.06 too high relative to the grid the curves were plotted on |
| `compare_pegbis.py` | Compares PEGBIS (`n_freq=10`) with `cal_surf_goda` on slopes 1/10–1/100 (paper Figs. 6–9) and computes Kweon's horizontal-step case (Fig. 11); writes `pegbis/compare_slope.png`, `pegbis/kweon_step.png` |
| `compare_pegbis_paper.py` | Overlays `cal_surf_pegbis` on the paper's p.19 付図-4〜7 and p.7 Fig. 11, calibrated on chart frames / gridlines of 300 dpi page images `pegbis/ecoh2-19.png`, `pegbis/ecoh2-07.png` (made with `pdftoppm` from the ECOH PDF; not committed); writes `pegbis/overlay_slope.png`, `pegbis/overlay_fig11.png` |
| `run.py` | Exploratory notebook-style plots (currently references undefined names and does not run as-is) |
| `draw_on_image.py` / `draw_on_image2.py` | Overlay results on reference images |

Run sample scripts from the `sample/` directory (they import sibling modules and load images via relative paths such as `../syuto.png`):
```bash
cd sample && uv run python check_goda.py
```

Samples are not covered by CI lint (`ruff` runs on `src/` and `test/` only).

Note: sample scripts use `matplotlib-fontja` for Japanese labels. This is a dev dependency.

---

## Git Branch Conventions

- Main development branch: `main`
- CI triggers on push/PR to `main`
- Feature branches follow the pattern: `claude/<description>-<session-id>`

---

## Do's and Don'ts for AI Assistants

**Do:**
- Run `uv run pytest -v` after any change to `core.py` or `constant.py`
- Run `uv run ruff check src/ test/` before committing
- Maintain type hints on all function signatures
- Keep helper functions in `core.py` co-located with the logic they support
- Use constants from `constant.py` — do not hardcode physical values inline

**Don't:**
- Add new top-level modules without strong justification
- Change the public API signature of `cal_surf_goda`, `cal_surf_goda_dim`, or `shoal` without updating tests
- Hardcode physical constants such as `g` or 2π — import them from `constant.py` (empirical model coefficients like 0.18 / 0.12 / 15 in `bindx()` or 1.416 in `prob()` currently live inline in `core.py`)
- Add dependencies to `pyproject.toml` without updating `uv.lock` via `uv sync`
- Skip linting — CI will fail on ruff errors

---

## References

- Goda, Y. (1975). Irregular wave deformation in the surf zone. *Coastal Engineering in Japan*, 18, 13–26.
- 合田良実 (1975). 浅海域における波浪の砕波変形. 港湾技術研究所報告, 14(3). ([PDF](https://www.pari.go.jp/PDF/vol014-no03-03.pdf))
- 合田良実 (2004). 不規則波浪変形計算における段階的砕波係数の導入について (Ver.5). ECOH/YG 技術論文 No.2. (https://www.ecoh.co.jp/goda/849/) — PEGBIS
- 合田良実 (2003). 段階的砕波変形モデルによる人工リーフの波浪減殺特性の解析 (Ver.3). ECOH/YG 技術論文 No.3. (https://www.ecoh.co.jp/goda/847/) — bottom friction, reef friction fw ≈ 0.15
- Project README (Japanese): `README.md`
- Technical skill sheet: `skill.md`
