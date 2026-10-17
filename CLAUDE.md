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
│       ├── core.py           # All computation logic (570 lines, 14 functions)
│       └── constant.py       # Physical constants & model parameters (142 lines)
├── test/
│   └── test_core.py          # Full test suite (390 lines, 56 test cases)
├── sample/                   # Usage examples and visualization scripts
│   ├── check_core.py
│   ├── check_goda.py
│   ├── check_shuto.py
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

Tests live in `test/test_core.py`. There are 56 test cases covering all public API functions. All tests must pass before committing.

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
| `cal_surf_goda_point` | `(tant, h0l0, dl0) → DataFrame` | Single-point result, computed by marching from offshore along `goda_dl0_list` so setup is included |

`cal_surf_goda` accepts `dl0` in any order: it sorts internally from offshore (largest) to onshore (smallest), marches setup in that direction, and returns rows in the caller's order. Duplicate `dl0` values raise `ValueError`. Setup is accumulated only over the points given, so a sparse `dl0` list gives a coarser setup estimate.
| `shoal` | `(dl0, h0l0) → float` | Shoaling coefficient (3-region model) |
| `aksi` | `(dl0) → float` | Linear (small-amplitude) shoaling coefficient |
| `cal_wave_length` | `(d, T) → float` | Wave length from depth and period |
| `wave` | `(d) → float` | Solve dispersion relation: x·tanh(x) = d |

### Output DataFrame Columns

| Column | Description |
|--------|-------------|
| `H1_1000` through `H1_3` | Representative wave heights (exceedance: 1/1000, 1/250, …, 1/3) |
| `dh0` | Depth-to-deep-water-height ratio |
| `dl0` | Depth-to-deep-water-wavelength ratio |
| `etal0` | Mean water level change ratio (η/L₀); 0 at the most offshore point, negative for set-down, positive for setup |
| `aks` | Shoaling coefficient |
| `d`, `H1_3_dim`, … | Dimensional values (only when `dim=True`) |

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

- Keep all computation logic in `src/goda_wb/core.py`
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
| `goda_dl0_list` | 114 values | Default d/L₀ evaluation points (0.00001–1.0) |
| `GAUSS_HERMITE_ABSCISSAE/WEIGHTS` | 8 points | Discretized normal distribution for surf beat (see note below) |

---

## Testing Conventions

- Tests are in `test/test_core.py`, using standard `pytest`
- Use `pytest.mark.parametrize` for boundary/edge case sweeps
- Use numerical regression tests (assert computed values ≈ expected reference values)
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

3. **Probability distribution**: `prob()` uses a Rayleigh distribution with breaking correction to compute the full wave height probability density across `MP=51` points. The breaking limits come from `bindx()` (coefficients 0.18 / 0.12).

4. **Surf beat**: `sbeat()` integrates over 8 surf-beat water level offsets. Despite the `GAUSS_HERMITE_*` names, the constants are a discretized standard normal distribution (weights 0.3413, 0.1359, … are band probabilities summing to 1), not Gauss–Hermite nodes/weights.

5. **Setup calculation**: `setup()` updates the mean water level by a single explicit step from the radiation-stress (energy) balance between point `i-1` and `i` — no iteration. `etai()` linearly extrapolates the previous two points to give the initial estimate used for breaking. This marching requires `dl0` ordered offshore → onshore, which `cal_surf_goda` guarantees by sorting.

---

## Sample Scripts

Located in `sample/` — not part of the installable package, intended for exploration and visualization:

| Script | Purpose |
|--------|---------|
| `check_core.py` | Basic API demonstration |
| `check_goda.py` | Wave transformation plots across steepness values |
| `check_shuto.py` | Shoaling coefficient curves (`shoal()`) for comparison with Shuto's shoaling chart |
| `run.py` | Exploratory notebook-style plots (currently references undefined names and does not run as-is) |
| `draw_on_image.py` / `draw_on_image2.py` | Overlay results on reference images |

Run sample scripts from the `sample/` directory (they import sibling modules and load images via relative paths such as `../syuto.png`):
```bash
cd sample && uv run python check_goda.py
```

Samples are not covered by CI lint (`ruff` runs on `src/` and `test/` only).

Note: sample scripts use `japanize-matplotlib` for Japanese labels. This is a dev dependency.

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
- Hardcode physical constants such as `g` or 2π — import them from `constant.py` (empirical model coefficients like 0.18 / 0.12 in `bindx()` or 1.416 in `prob()` currently live inline in `core.py`)
- Add dependencies to `pyproject.toml` without updating `uv.lock` via `uv sync`
- Skip linting — CI will fail on ruff errors

---

## References

- Goda, Y. (1975). Irregular wave deformation in the surf zone. *Coastal Engineering in Japan*, 18, 13–26.
- 合田良実 (1975). 浅海域における波浪の砕波変形. 港湾技術研究所報告, 14(3). ([PDF](https://www.pari.go.jp/PDF/vol014-no03-03.pdf))
- Project README (Japanese): `README.md`
- Technical skill sheet: `skill.md`
