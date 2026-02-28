# CLAUDE.md — AI Assistant Guide for `goda-wb`

This file provides context for AI assistants (Claude Code and others) working on this repository.

---

## Project Overview

**goda-wb** is a Python library implementing Goda's breaking wave deformation model for coastal engineering. It computes wave transformation in the surf zone, accounting for wave breaking, shoaling, and wave setup using probabilistic wave height distributions.

- **Primary reference**: Goda (1975) — irregular wave breaking and surf zone hydrodynamics
- **Author**: tlasu (sukazuki.mail@gmail.com)
- **License**: MIT
- **Version**: 0.1.0

---

## Repository Structure

```
goda_wb/
├── src/
│   └── goda_wb/
│       ├── __init__.py       # Package init — re-exports `core`
│       ├── core.py           # All computation logic (551 lines, 18 functions)
│       └── constant.py       # Physical constants & model parameters (145 lines)
├── test/
│   └── test_core.py          # Full test suite (334 lines, 50 test cases)
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

Tests live in `test/test_core.py`. There are 50 test cases covering all public API functions. All tests must pass before committing.

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

All functions are in `goda_wb.core` and re-exported from `goda_wb`.

### Primary Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `cal_surf_goda` | `(tant, h0l0, dl0=None) → DataFrame` | Main dimensionless wave transformation calculation |
| `cal_surf_goda_dim` | `(tant, H0, T, dim=False, d=None) → DataFrame` | Dimensional wrapper (SI units) |
| `cal_surf_goda_point` | `(tant, h0l0, dl0) → DataFrame` | Single-point calculation |
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
| `etal0` | Setup height ratio (η/L₀) |
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
- Internal/helper functions are not prefixed but are grouped at the bottom of `core.py`
- Trailing underscores (`etl0_`, `h2l0_`) denote intermediate computed values

### Module Organization

- Keep all computation logic in `src/goda_wb/core.py`
- Keep all constants (physical, numerical, model) in `src/goda_wb/constant.py`
- Do not add new modules without clear necessity — the codebase is intentionally compact

### Style

- Line length: 100 characters (enforced by ruff)
- Type hints are used throughout — maintain them in any additions
- Docstrings are in English
- No external side effects — all functions are stateless and pure

### Constants (from `constant.py`)

| Name | Value | Purpose |
|------|-------|---------|
| `g` | 9.81 | Gravitational acceleration (m/s²) |
| `pi2` | 2π | Convenience constant |
| `MP` | 51 | Discretization points for probability density |
| `NWAVE` | 7 | Number of representative wave heights |
| `CONVERGENCE_TOL` | 1e-6 | Newton method convergence tolerance |
| `WAVE_CONVERGENCE_TOL` | 0.0003 | Wave dispersion iteration tolerance |
| `DEEP_WATER_THRESHOLD` | 10.0 | dl0 threshold for deep water classification |
| `goda_dl0_list` | 114 values | Default d/L₀ evaluation points (0.00001–1.0) |
| `GAUSS_HERMITE_ABSCISSAE/WEIGHTS` | 8-point | Quadrature for surf beat integration |

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

1. **Wave dispersion**: `wave(d)` solves x·tanh(x) = d iteratively (Newton's method), converging to `WAVE_CONVERGENCE_TOL`.

2. **Shoaling model**: `shoal()` uses a 3-region model:
   - Deep water: `aks = 1.0`
   - Intermediate: computed from group velocity ratio
   - Shallow water: matched to deep water limit

3. **Probability distribution**: `prob()` uses a Rayleigh distribution with breaking correction to compute the full wave height probability density across `MP=51` points.

4. **Gauss-Hermite quadrature**: Used in `sbeat()` for surf beat water level variation integration (8-point rule, pre-computed nodes/weights in `constant.py`).

5. **Setup calculation**: `setup()` uses energy flux conservation with iterative Newton convergence (`CONVERGENCE_TOL = 1e-6`).

6. **Stateless design**: All functions take parameters and return values — no global state is modified.

---

## Sample Scripts

Located in `sample/` — not part of the installable package, intended for exploration and visualization:

| Script | Purpose |
|--------|---------|
| `check_core.py` | Basic API demonstration |
| `check_goda.py` | Wave transformation plots across steepness values |
| `check_shuto.py` | Storm surge scenario analysis |
| `run.py` | Full analysis with matplotlib output |
| `draw_on_image.py` / `draw_on_image2.py` | Overlay results on reference images |

Run sample scripts with:
```bash
uv run python sample/check_goda.py
```

Note: sample scripts use `japanize-matplotlib` for Japanese labels. This is a dev dependency.

---

## Git Branch Conventions

- Main development branch: `master`
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
- Use floating point literals for physical constants — refer to `constant.py`
- Add dependencies to `pyproject.toml` without updating `uv.lock` via `uv sync`
- Skip linting — CI will fail on ruff errors

---

## References

- Goda, Y. (1975). Irregular wave deformation in the surf zone. *Coastal Engineering in Japan*, 18, 13–26.
- Goda (1970): Original irregular wave model
- Project README (Japanese): `README.md`
- Technical skill sheet: `skill.md`
