"""Shared helpers for the Jupiter GCM demo notebooks.

Both the Lian-Showman relaxation model and the grey-radiation model build
their grid, initial conditions, time stepper, and output-writing the same
way; this module factors that out so the notebooks can focus on the physics
that differs between them.
"""

import functools
import time

import dinosaur
import jax
import numpy as np
import xarray

units = dinosaur.scales.units


def dimensionalize(
    physics_specs: dinosaur.units.SimUnitsProtocol,
    x: xarray.DataArray,
    unit: units.Unit,
) -> xarray.DataArray:
  """Dimensionalizes an `xarray.DataArray` of nondimensional values."""
  return xarray.apply_ufunc(
      functools.partial(physics_specs.dimensionalize, unit=unit), x
  )


def build_coords(
    layers: int = 60,
    scale_heights: int = 11,
    horizontal_grid: str = 'T42',
) -> 'dinosaur.coordinate_systems.CoordinateSystem':
  """Builds a Jupiter-appropriate coordinate system.

  Args:
    layers: number of vertical sigma levels.
    scale_heights: number of log-sigma scale heights spanned by the vertical
      grid (see `sigma_coordinates.SigmaCoordinates.equidistant_log`); a deep
      atmosphere like Jupiter's needs more scale heights than Earth to
      resolve both the upper atmosphere and the deep interior.
    horizontal_grid: horizontal spectral truncation, e.g. `'T42'`.

  Returns:
    A `CoordinateSystem` with no `spmd_mesh` set (single-device use).
  """
  horizontal = getattr(dinosaur.spherical_harmonic.Grid, horizontal_grid)()
  vertical = dinosaur.sigma_coordinates.SigmaCoordinates.equidistant_log(
      layers, scale_heights
  )
  return dinosaur.coordinate_systems.CoordinateSystem(
      horizontal=horizontal, vertical=vertical
  )


def initial_state(
    coords: 'dinosaur.coordinate_systems.CoordinateSystem',
    physics_specs: 'dinosaur.units.SimUnitsProtocol',
    p0: 'units.Quantity',
    p1: 'units.Quantity' = 5e3 * units.pascal,
    seed: int = 0,
):
  """Returns an initial `State`, reference temperatures, and orography.

  Uses an isothermal rest atmosphere with a small random surface-pressure
  perturbation, matching the standard Held-Suarez initialization approach.
  """
  initial_state_fn, aux_features = (
      dinosaur.primitive_equations_states.isothermal_rest_atmosphere(
          coords=coords, physics_specs=physics_specs, p0=p0, p1=p1
      )
  )
  rng_key = jax.random.PRNGKey(seed)
  state = initial_state_fn(rng_key)
  ref_temps = aux_features[dinosaur.xarray_utils.REF_TEMP_KEY]
  orography = dinosaur.primitive_equations.truncated_modal_orography(
      aux_features[dinosaur.xarray_utils.OROGRAPHY], coords
  )
  return state, ref_temps, orography


def trajectory_to_xarray(
    coords: 'dinosaur.coordinate_systems.CoordinateSystem',
    physics_specs: 'dinosaur.units.SimUnitsProtocol',
    trajectory,
    times: np.ndarray,
    ref_temps: np.ndarray,
) -> xarray.Dataset:
  """Converts a raw model trajectory into a dimensionalized `xarray.Dataset`."""
  dimensionalize_ = functools.partial(dimensionalize, physics_specs)

  trajectory_dict, _ = dinosaur.pytree_utils.as_dict(trajectory)
  u, v = dinosaur.spherical_harmonic.vor_div_to_uv_nodal(
      coords.horizontal, trajectory.vorticity, trajectory.divergence
  )

  trajectory_dict.update({
      'u': dimensionalize_(u, units.meter / units.second),
      'v': dimensionalize_(v, units.meter / units.second),
      'vorticity': dimensionalize_(trajectory.vorticity, 1 / units.second),
      'divergence': dimensionalize_(trajectory.divergence, 1 / units.second),
  })
  nodal_fields = dinosaur.coordinate_systems.maybe_to_nodal(
      trajectory_dict, coords=coords
  )
  ds = dinosaur.xarray_utils.data_to_xarray(
      nodal_fields, coords=coords, times=times
  )

  ds['surface_pressure'] = dimensionalize_(
      np.exp(ds.log_surface_pressure[:, 0, :, :]), units.pascal
  )
  temperature = dinosaur.xarray_utils.temperature_variation_to_absolute(
      ds.temperature_variation.data, ref_temps
  )
  ds = ds.assign(
      temperature=(
          ds.temperature_variation.dims,
          dimensionalize_(temperature, units.degK),
      )
  )

  total_layer_ke = coords.horizontal.integrate(u**2 + v**2)
  total_ke_cumulative = dinosaur.sigma_coordinates.cumulative_sigma_integral(
      total_layer_ke, coords.vertical, axis=-1
  )
  ds = ds.assign(
      total_kinetic_energy=(
          ('time',),
          dimensionalize_(
              total_ke_cumulative[..., -1],
              units.meter**2 / units.second**2,
          ),
      )
  )
  return ds


def run_integration(
    equations,
    coords: 'dinosaur.coordinate_systems.CoordinateSystem',
    physics_specs: 'dinosaur.units.SimUnitsProtocol',
    initial_state,
    ref_temps: np.ndarray,
    dt_si: 'units.Quantity' = 10 * units.minute,
    save_every: 'units.Quantity' = 1 * units.day,
    total_time: 'units.Quantity' = 1 * units.day,
    filter_tau: float = 0.0087504,
    filter_order: float = 1.5,
    filter_cutoff: float = 0.8,
):
  """Runs a short forward integration and returns a dimensionalized dataset.

  `equations` should be a single equation system or a list to be combined
  with `time_integration.compose_equations` (e.g.
  `[primitive_equations, forcing]`).
  """
  if isinstance(equations, (list, tuple)):
    equations = dinosaur.time_integration.compose_equations(list(equations))

  dt = physics_specs.nondimensionalize(dt_si)
  inner_steps = int(save_every / dt_si)
  outer_steps = int(total_time / save_every)

  step_fn = dinosaur.time_integration.imex_rk_sil3(equations, dt)
  filters = [
      dinosaur.time_integration.exponential_step_filter(
          coords.horizontal,
          dt,
          tau=filter_tau,
          order=filter_order,
          cutoff=filter_cutoff,
      ),
  ]
  step_fn = dinosaur.time_integration.step_with_filters(step_fn, filters)

  integrate_fn = jax.jit(
      dinosaur.time_integration.trajectory_from_step(
          step_fn, outer_steps=outer_steps, inner_steps=inner_steps
      )
  )

  times = save_every * np.arange(1, outer_steps + 1)

  print(f'Running on device(s): {jax.devices()}')
  start_time = time.time()
  final_state, trajectory = jax.block_until_ready(integrate_fn(initial_state))
  elapsed = time.time() - start_time
  print(f'Integration took {elapsed:.1f}s for {outer_steps * inner_steps} steps')

  ds = trajectory_to_xarray(
      coords, physics_specs, jax.device_get(trajectory), times, ref_temps
  )
  return final_state, ds, elapsed
