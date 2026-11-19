# Differentiable Ship-D in JAX (`shipd_jax`)

This package started as the standalone `shipd_jax`; that name still appears
below for the JAX port as opposed to the original implementation, which is
kept, compiled with Cython, in `shipd_jax.reference`.

A JAX re-implementation of the Ship-D hull parameterization
(`HullParameterization.py`, Bagazinski & Ahmed 2023) in which the **hull
surface, all 49 feasibility constraints, the hydrostatic measures, and the
Michell wave resistance are differentiable** with respect to the 45-entry
design vector — while producing near-identical numerical output to the
original NumPy implementation.

Validated agreement with the reference (12 sample hulls + 25 hulls spanning
the 30k dataset + one run at full dataset resolution, 101x1000; worst case
across all hulls — rerun `tools/validate_vs_reference.py` to reproduce):

| quantity                                   | worst-case difference |
|--------------------------------------------|-----------------------|
| 49 constraints                             | 2.8e-15 (abs)         |
| measurement point cloud (`gen_pointCloud`) | 1.9e-9 (abs)          |
| volumes / areas / centers / moments / WSA  | 3.8e-11 (rel)         |
| Michell inputs (`gen_PC_for_Cw` grid)      | 3.7e-9 (abs)          |
| Michell wave resistance                    | 3.9e-10 (rel)         |
| STL mesh (`gen_stl`)                       | bit-exact at float32  |

Gradients (reverse and forward mode) are finite and match central finite
differences of the forward model to ~1e-10 (volume, wetted surface,
constraints, surface points) and ~1e-6 for the Michell resistance (FD
truncation noise on an oscillatory integral, not AD error).

One caveat inherited from the original formulation: several constraints
branch on exact conditions like ``DELTA_STERN[0] == 0`` and are therefore
*discontinuous* in the design variables at those measure-zero sets (e.g.
hull `csv[3]` sits exactly on one).  AD returns the correct derivative of
the active branch; a finite difference stepping across the jump will
disagree there — in the original NumPy code the same jump exists.

## How it works: plans (static topology) + traced coordinates

The original code makes discrete decisions from the concrete parameter
values: how many points each waterline section gets, which cubic root is the
bulb–hull intersection, ellipse vs. parabola for the stern waterline, etc.
These are piecewise-constant in the design variables, so they cannot (and
should not) be differentiated through. `shipd_jax` splits the computation:

* a **plan** (`build_volume_plan`, `build_cw_plan`) computes all such
  structure once from a concrete numpy vector, by evaluating the *same* JAX
  functions eagerly;
* all coordinates/values are then recomputed **traced** from the JAX design
  vector with the plan's structure held fixed.

With the plan built from the same vector you evaluate at, forward results
match the original implementation; gradients are the shape sensitivities at
fixed mesh topology. If you move the design point far from where the plan
was built, rebuild the plan (cheap, pure numpy speed).

The bulb–hull intersection root (found with `np.roots` in the original) is
reproduced by Newton refinement from a plan-stored initializer, so it is exact
at the evaluation point and its gradient is the implicit-function derivative.

## Usage

```python
import numpy as np, jax, jax.numpy as jnp
from shipd_jax import SAMPLE_HULLS
from shipd_jax import (bits_from_vector, constraints,
                                     build_volume_plan, gen_pointcloud, volume_properties,
                                     build_cw_plan, gen_pc_for_cw, mod_michell, calc_cf)

x = np.loadtxt(SAMPLE_HULLS, delimiter=',')[0]
bits = bits_from_vector(x)          # static 0/1 switches (EP_S, EP_T, BB, SB)

# ---- constraints (feasible: all <= 0) and their jacobian ----
c  = constraints(x, bits)                                # (49,)
Jc = jax.jacfwd(lambda v: constraints(v, bits))(jnp.asarray(x))   # (49, 45)

# ---- hull surface and its sensitivity ----
plan = build_volume_plan(x, num_wl=101, ppw=1000)
pc   = gen_pointcloud(jnp.asarray(x), plan)              # (101, 1000, 3)
dpc  = jax.jacfwd(lambda v: gen_pointcloud(v, plan))(jnp.asarray(x))

# ---- hydrostatics (Volumes, Areas_WP, LCF, LCB/VCB, Ixx/Iyy, WSA, WL length) ----
props = volume_properties(jnp.asarray(x), plan)
dV = jax.grad(lambda v: volume_properties(v, plan)['volumes'][-1])(jnp.asarray(x))

# ---- Michell wave resistance at draft T = 0.33*Dd, Fn = 0.3 ----
cw_plan = build_cw_plan(x, draft_frac=0.33, num_wl=51, ppw=301)

def rw(v, Fn=0.3):
    X, Z, Y, WL = gen_pc_for_cw(v, cw_plan)
    U = Fn * jnp.sqrt(9.81 * WL)
    return mod_michell(Y, U, X, Z, rho=1000.0, N=21)

value = rw(jnp.asarray(x))
grad  = jax.grad(rw)(jnp.asarray(x))
```

Everything composes with `jax.jit` (close over the plan):

```python
rw_fast = jax.jit(rw)
```

### STL export

```python
from shipd_jax.stl_export import gen_stl
gen_stl(x, NUM_WL=50, PointsPerWL=300, bit_AddTransom=1, bit_AddDeckLid=1,
        namepath='My_Hull')
```

This reproduces the reference mesh exactly (same triangulation, same
vertices). Mesh assembly indices are concrete by nature, so STL export is a
convenience output rather than a differentiable path; the differentiable
surface is `gen_pointcloud` / `gen_pc_for_cw`.

## Validation

```bash
python tools/validate_vs_reference.py            # sample + 20 hulls of the 30k set
python tools/validate_vs_reference.py --full-res # adds a 101x1000 dataset-resolution check
pytest tests/test_stl_parity.py                  # STL mesh comparison
```

The validator compares every quantity against the original implementation
(`shipd_jax.reference`) and checks JAX gradients against central
finite differences of the JAX forward. The 30k-hull check reads
`InputVectors_30k.npy` (the ShipD dataset, not committed) and is
skipped without it.

The numbers above were produced against a pure-Python copy of the original
(`spring_26_rewrite/HullParameterization.py`, not in this repository). The
compiled reference must be built with Cython's `wraparound` left on (see
`setup.py`): the code indexes from the end of arrays.

## Faithfully reproduced quirks

To achieve numerical parity, the following behaviors of the reference are
reproduced on purpose:

* `GenBowForm` tests the bow-rise vertex against `BK[1]*Dd` although `BK[1]`
  is already dimensional (a double scaling);
* both fillet quartic systems use `6*x**2` in the second-derivative row
  (instead of `6*x`);
* `Calc_WettedSurface` contains a no-op line where the transom width was
  meant to be added to the arc length — it is therefore *not* added;
* in the Michell integral, propagation angles near pi/2 whose Filon weights
  overflow `exp` produce NaN in the reference and are zeroed by
  `np.nan_to_num`; here they are masked to exactly zero contribution (same
  value, finite gradients);
* if a Cw grid station coincides exactly with `delta_bow(z)` (which happens
  systematically when the delta-bow curve is constant), the reference
  double-counts that station and its Y-grid assembly shifts by one slot; the
  plan detects this and replicates the shifted assembly.

Known reference-side failures are *not* worked around: e.g. hulls whose
stern-bulb fillet system is singular (`csv[8]` in the sample vectors) crash
the original `gen_waterline_stern_SB`; here they produce non-finite values
instead of raising.

## Optimizer readiness: the plan-free pipeline (`shipd_jax.universal_full`)

The reference-exact pipeline freezes discrete mesh topology (per-waterline
integer point counts, curve-family flags, bulb-fillet root initializers) into
a plan built at one design vector. Those integers are **array shapes**, so the
plan is baked into the JIT'd executable — and it goes stale almost immediately
as an optimizer moves (measured: 7/41 waterlines wrong at ‖Δx‖≈0.0015,
all 41 by ‖Δx‖≈0.015). Rebuilding it forces a 30–170 s XLA recompile.

`shipd_jax.universal_full` removes the dependency for **every** hull type
(supersedes an earlier bulb-free-only version, `shipd_jax.universal`, which
handled only `bit_EP_S == bit_EP_T` and has since been removed):

* `section_counts(ppw, bow_frac, stern_frac)` — point counts are fixed
  fractions of `ppw`, valid for any design vector. Sample *locations* still
  adapt continuously; only the *counts* are frozen.
* `universal_pc_for_cw(x, draft_frac, bits, ...)` — the offset-grid mask
  becomes a traced `jnp.where`, and **the draft is a traced argument**, so one
  compile covers every (design, draft) pair and `dCw/d(draft)` exists.
* `universal_pointcloud(x, bits, ...)` / `universal_volume_properties(x, bits, ...)`
  — the plan-free hull surface and hydrostatics, used by
  `ship_mdo.models.shipd_resistance.ShipDResistanceJAX`, the CSDL bridge in
  [ship-mdo](https://github.com/jer-j/ship-mdo) that its optimization notebooks run.

The key realization over the bulb-free-only version: the reference plan
freezes two *different* kinds of thing, and only array **shapes** genuinely
have to be static. "Is the bulb present at this waterline?", "ellipse or
parabola?", "which fillet-cubic root?" etc. are *value* decisions expressible
with `jax.lax.cond`/`jnp.where` from the traced design vector — `lax.cond`
specifically (not `where`) for the bulb branches, since the untaken branch
would otherwise evaluate bulb geometry outside its valid domain and poison
the gradient with NaN even though `where` selects the right forward value.
One compiled executable then serves every design vector per `Bits` setting
(at most 16 variants, structural switches rather than design variables), with
branch decisions moving *with* the design instead of being frozen at plan-build
time — so gradients stay correct arbitrarily far from the point the topology
was chosen at.

Validated (bulb-free case, against the reference plan): the Michell offset
grid is **bit-identical** (0.0e+00, so no wave-drag penalty); hydrostatics
differ only by sample placement and converge with resolution (4.7e-5 at
21×101 → 4.5e-6 at 51×300); `dCw/d(draft)` matches finite differences to
2e-8; and design gradients stay correct **at points where the frozen plan
would be stale** (5.4e-7 vs FD). Also separately validated bit-identical
(~1e-14) against the reference for a **bulbous** hull, including at points
far outside the baseline used to build the reference plan (see
`examples/ShipD_JAX_Universal_Validation.ipynb`).

`shipd_jax.iges128` is a small reader for IGES entity-128 rational B-spline
surfaces, used to compare against published hull geometry (e.g. the SIMMAN
2008 DTMB 5415 IGES). It downloads nothing; callers fetch files themselves.

## Visualization (`shipd_jax.viz`)

Every plot in the analysis notebook is a call into `shipd_jax.viz`
(also available as `viz` after `import shipd_jax`), so any of
it is reusable outside the notebook. Four families of functions:

* **Resistance curves**: `plot_michell_validation`, `plot_resistance_vs_froude`
  (Figure-7-style multi-draft curve + tank data + reference overlay),
  `plot_dcw_dfn` (dual-axis Cw and dCw/dFn with humps/hollows marked).
* **Sensitivity diagnostics**: `plot_param_gradient_bars` (one or more 45-vectors
  as grouped bars), `plot_constraint_jacobian` (49x45 heatmap, grouped by
  constraint family), `kkt_matrix`/`hessian_definiteness`/`plot_kkt_matrices`
  (assemble and render `[[H, J_A^T],[J_A, 0]]` panels across e.g. several
  Froude numbers).
* **3D rendering**: `plot_hull_3d`, `plot_hull_surface_scalar` (generic
  scalar-colored surface), `plot_waterlines_plan`, `plot_chamfer_comparison`.
* **Traditional lines drawings**: `plot_profile_plan` (keel/stem/sternpost
  outline + buttock lines), `plot_halfbreadth_plan` (waterlines, mirrored),
  `plot_body_plan` (stations, forward on the right / aft on the left by
  convention, `flip=True` to mirror), and `plot_lines_drawing` (all three
  composed from one `gen_pointcloud` call). Pure numpy post-processing of the
  point grid — no new differentiable geometry needed, since every waterline
  row is already x-monotonic bow-to-stern.
* **Shape sensitivity**: `plot_param_sensitivity_3d(x, plan, param)` colors the
  hull surface by `‖∂p/∂x_i‖` for any design variable *except* the four
  discrete bit switches (raises `ValueError` — they have no derivative).
  `plot_objective_sensitivity_3d(x, plan, objective_fn)` colors the surface by
  the steepest-ascent shape-deformation field of a scalar objective — build
  `objective_fn` with `make_resistance_objective(cw_plan, Fn, mode='Rw'|'Cw')`
  or `make_constraint_objective(bits, k)` for any single constraint. Both are
  thin wrappers around `surface_sensitivity_to_param`/`_to_objective`
  (`jax.jvp`/`jax.grad` + `jax.jvp`) if you want the raw arrays instead of a
  plot. These convenience wrappers are un-jitted (simple, one-off use); for a
  sweep over many variables/objectives, jit the surface-JVP kernel yourself
  and reuse it — see Section 9 of the notebook for the pattern.

## Analysis notebook

`examples/ShipD_JAX_Analysis.ipynb` reproduces Figures 6 and 7 of the Ship-D paper
(Wigley Michell validation — essentially exact; DTMB 5415 vs. towing tank),
and demonstrates the differentiable pipeline end to end: `dCw/dFn` via
`value_and_grad`, resistance/constraint Jacobians and exact-autodiff **KKT
matrices** at several Froude numbers, the package demo, 3D hull surfaces and
waterline plan views, the paper's Chamfer metric (showing `shipd_jax` matches
the original implementation to ~1e-13 % of LOA), **traditional lines drawings
and body plans**, and **surface-sensitivity fields for any design variable,
for resistance ($R_w$/$C_w$), or for any constraint** — the last two sharing a
single compiled surface-JVP kernel reused across every direction examined.
`data/paper_data/` holds the extracted paper figures and their digitized curves.
The Michell y-axis convention used by the paper (recovered by matching the
analytic Wigley curve) is `Cw = Rw / (rho U^2 L^2)` — no factor 1/2.

## Files

Library modules are in `shipd_jax/`, notebooks in `examples/`, scripts in
`tools/`, tests in `tests/`. The original `HullParameterization.py` and
`ModifiedMichellCw.py` at the repository root are upstream's, untouched;
`shipd_jax/reference/` holds the Cython-compiled copies the port is checked
against (`hull_parameterization.pyx` differs from `HullParameterization.py`
only by one commented-out debug `print`).

| file | contents |
|---|---|
| `hull.py` | parameter parsing, all sectional geometry, fillets, plans, point clouds, hydrostatics |
| `universal_full.py` | plan-free (recompile-free) pipeline for **every** hull type; supersedes the removed bulb-free-only `universal.py` |
| `viz.py` | reusable visualizations: resistance curves, gradient/Jacobian/KKT diagnostics, 3D rendering, lines drawings, shape sensitivity |
| `iges128.py` | minimal IGES entity-128 NURBS reader (for comparing against published geometry) |
| `ShipD_JAX_Analysis.ipynb` | paper reproduction + differentiability showcase (see above) |
| `ShipD_JAX_Optimizer_Readiness.ipynb` | recompilation study (bounds / `lax.cond` / plan-free rewrite), sensitivity validation, Fig. 7 draft reverse-engineering, Chamfer audit vs. the official SIMMAN 5415 -- historical: some cells exercise the now-removed `universal.py`, kept for their recorded results/rationale rather than re-run |
| `ShipD_JAX_Universal_Validation.ipynb` | validates `universal_full.py` (bulb-free and bulbous cases) bit-identical against the reference plan pipeline |
| `constraints.py` | the 49 feasibility constraints |
| `michell.py` | vectorized Michell integral, ITTC friction line, drag/interp helpers |
| `stl_export.py` | reference-exact STL mesh generation (concrete) |
| `reference/` | the original implementation (Cython), the reference the above is checked against |
| `tools/validate_vs_reference.py` | numerical comparison + gradient checks |
| `tests/test_stl_parity.py` | STL mesh comparison |

Requires `jax` (x64 enabled at import), `numpy`. Tested with jax 0.4.38 /
numpy 1.26.4 in the `mdo_env` conda environment.
