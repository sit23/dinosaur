"""Shared helpers for the Jupiter GCM demo notebooks.

Both the Lian-Showman relaxation model and the grey-radiation model build
their grid, initial conditions, time stepper, and output-writing the same
way; this module factors that out so the notebooks can focus on the physics
that differs between them.
"""

import functools
import glob
import json
import os
import time

import dinosaur
import jax
import jax.numpy as jnp
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


def build_model_equations(
    model_name: str,
    coords: 'dinosaur.coordinate_systems.CoordinateSystem',
    physics_specs: 'dinosaur.units.SimUnitsProtocol',
    ref_temps: np.ndarray,
    orography,
    p0: 'units.Quantity',
):
  """Builds the `[primitive, forcing(s)]` equations list for a named model."""
  primitive = dinosaur.primitive_equations.PrimitiveEquations(
      ref_temps, orography, coords, physics_specs
  )
  if model_name == 'lian_showman':
    forcing = dinosaur.held_suarez.LianShowmanForcing(
        coords=coords,
        physics_specs=physics_specs,
        reference_temperature=ref_temps,
        p0=p0,
    )
    return [primitive, forcing]
  elif model_name == 'grey_radiation':
    radiation = dinosaur.grey_radiation.GiantPlanetGreyRadiation(
        coords=coords,
        physics_specs=physics_specs,
        reference_temperature=ref_temps,
        p0=p0,
    )
    convection = dinosaur.dry_convection.DryConvectiveAdjustment(
        coords=coords,
        physics_specs=physics_specs,
        reference_temperature=ref_temps,
    )
    return [primitive, radiation, convection]
  else:
    raise ValueError(f'unknown model_name {model_name!r}')


def benchmark_resolution(
    model_name: str,
    max_wavenumber: int,
    gaussian_nodes: int,
    dt_si: 'units.Quantity',
    p0: 'units.Quantity' = 25e5 * units.pascal,
    layers: int = 60,
    scale_heights: int = 11,
    benchmark_steps: int = 30,
    filter_tau: float = 0.0087504,
    filter_order: float = 1.5,
    filter_cutoff: float = 0.8,
) -> dict:
  """Benchmarks ms/step for one model at one horizontal resolution.

  Times a single compiled `benchmark_steps`-step advance, called twice: the
  first call triggers JIT compilation (untimed), the second reuses the
  compiled executable and is what's timed -- so this measures steady-state
  execution throughput, not one-off compile cost.

  Returns a dict of results; on failure (e.g. GPU OOM at high resolution)
  `result['error']` is set and `result['ms_per_step']` is `None` -- callers
  doing a resolution sweep should stop escalating resolution when this
  happens rather than trying successively larger (and slower-to-fail)
  resolutions.
  """
  result = {
      'model': model_name,
      'max_wavenumber': max_wavenumber,
      'gaussian_nodes': gaussian_nodes,
      'dt_si_seconds': dt_si.to(units.second).magnitude,
      'ms_per_step': None,
      'error': None,
  }
  try:
    horizontal = dinosaur.spherical_harmonic.Grid.construct(
        max_wavenumber=max_wavenumber, gaussian_nodes=gaussian_nodes
    )
    vertical = dinosaur.sigma_coordinates.SigmaCoordinates.equidistant_log(
        layers, scale_heights
    )
    coords = dinosaur.coordinate_systems.CoordinateSystem(
        horizontal=horizontal, vertical=vertical
    )
    result['longitude_nodes'] = horizontal.longitude_nodes
    result['latitude_nodes'] = horizontal.latitude_nodes

    physics_specs = (
        dinosaur.primitive_equations.PrimitiveEquationsSpecs.from_si()
    )
    state, ref_temps, orography = initial_state(coords, physics_specs, p0=p0)
    equations = dinosaur.time_integration.compose_equations(
        build_model_equations(
            model_name, coords, physics_specs, ref_temps, orography, p0
        )
    )

    dt = physics_specs.nondimensionalize(dt_si)
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
    advance_fn = jax.jit(
        dinosaur.time_integration.trajectory_from_step(
            step_fn, outer_steps=1, inner_steps=benchmark_steps
        )
    )

    jax.block_until_ready(advance_fn(state))  # compile, untimed
    start = time.time()
    jax.block_until_ready(advance_fn(state))  # timed, reuses compiled cache
    elapsed = time.time() - start

    result['ms_per_step'] = elapsed / benchmark_steps * 1000
    result['benchmark_wall_seconds'] = elapsed
  except Exception as e:  # pylint: disable=broad-except
    result['error'] = f'{type(e).__name__}: {e}'
  return result


def implied_wall_time(
    result: dict, total_time_si: 'units.Quantity' = 3 * units.day
) -> float | None:
  """Implied wall-clock seconds to simulate `total_time_si` at this resolution.

  Extrapolated from `benchmark_resolution`'s steady-state ms/step, rather
  than actually run to `total_time_si` -- at high resolution the step count
  needed (with a CFL-limited timestep) makes a literal run impractically
  slow, and per-step cost is stable enough that a short benchmark is
  representative.
  """
  if result.get('ms_per_step') is None:
    return None
  n_steps = int(
      np.ceil(total_time_si.to(units.second).magnitude / result['dt_si_seconds'])
  )
  return result['ms_per_step'] * n_steps / 1000


def resolution_scaling_sweep(
    model_name: str,
    resolutions: list[tuple[int, int, 'units.Quantity']],
    p0: 'units.Quantity' = 25e5 * units.pascal,
    layers: int = 60,
    scale_heights: int = 11,
    total_time_si: 'units.Quantity' = 3 * units.day,
    benchmark_steps: int = 30,
) -> list[dict]:
  """Benchmarks `model_name` over increasing resolutions, stopping on failure.

  Args:
    model_name: 'lian_showman' or 'grey_radiation'.
    resolutions: `(max_wavenumber, gaussian_nodes, dt_si)` tuples in
      increasing-resolution order; `dt_si` should be halved each time
      resolution doubles to keep the CFL number roughly constant.
    total_time_si: simulated duration each resolution's implied wall time is
      extrapolated to.
    benchmark_steps: number of steps actually timed at each resolution.

  Returns:
    List of result dicts, one per resolution attempted (including the first
    failure, if any, with `error` set and `ms_per_step`/implied time `None`).
  """
  results = []
  for max_wavenumber, gaussian_nodes, dt_si in resolutions:
    print(
        f'{model_name}: max_wavenumber={max_wavenumber}, '
        f'gaussian_nodes={gaussian_nodes}, dt={dt_si}'
    )
    result = benchmark_resolution(
        model_name,
        max_wavenumber,
        gaussian_nodes,
        dt_si,
        p0=p0,
        layers=layers,
        scale_heights=scale_heights,
        benchmark_steps=benchmark_steps,
    )
    result['implied_wall_seconds'] = implied_wall_time(result, total_time_si)
    results.append(result)
    if result['error'] is not None:
      print(f'  FAILED: {result["error"]}')
      print('  stopping sweep: further resolutions would likely also fail')
      break
    print(
        f'  ms/step={result["ms_per_step"]:.2f}, '
        f'implied wall time={result["implied_wall_seconds"]:.1f}s'
    )
  return results


def _state_to_numpy_dict(
    state: 'dinosaur.primitive_equations.State',
) -> dict[str, np.ndarray]:
  return {
      'vorticity': np.asarray(jax.device_get(state.vorticity)),
      'divergence': np.asarray(jax.device_get(state.divergence)),
      'temperature_variation': np.asarray(
          jax.device_get(state.temperature_variation)
      ),
      'log_surface_pressure': np.asarray(
          jax.device_get(state.log_surface_pressure)
      ),
  }


def save_checkpoint(path: str, state, metadata: dict) -> None:
  """Atomically saves raw model state + run metadata to `path` (a .npz file).

  Writes to `{path}.tmp` then `os.replace`s it into place, so a process
  killed mid-write leaves the previous checkpoint (if any) intact rather
  than a corrupt one.
  """
  arrays = _state_to_numpy_dict(state)
  tmp_path = f'{path}.tmp'
  # `np.savez` silently appends `.npz` to string paths that don't already end
  # in it (including `...npz.tmp`), which would break the atomic-rename
  # below -- pass a file handle instead, which it writes to as-is.
  with open(tmp_path, 'wb') as f:
    np.savez(f, metadata_json=json.dumps(metadata), **arrays)
  os.replace(tmp_path, path)


def load_checkpoint(
    path: str,
) -> tuple['dinosaur.primitive_equations.State', dict]:
  """Loads a checkpoint saved by `save_checkpoint`. Returns `(state, metadata)`."""
  data = np.load(path)
  state = dinosaur.primitive_equations.State(
      vorticity=jnp.asarray(data['vorticity']),  # pyrefly: ignore[unexpected-keyword]
      divergence=jnp.asarray(data['divergence']),  # pyrefly: ignore[unexpected-keyword]
      temperature_variation=jnp.asarray(  # pyrefly: ignore[unexpected-keyword]
          data['temperature_variation']
      ),
      log_surface_pressure=jnp.asarray(  # pyrefly: ignore[unexpected-keyword]
          data['log_surface_pressure']
      ),
  )
  metadata = json.loads(data['metadata_json'].item())
  return state, metadata


def _exact_ratio(numerator: float, denominator: float, num_name: str, den_name: str) -> int:
  """Returns `numerator / denominator` as an int, or raises if not exact."""
  ratio = numerator / denominator
  if abs(ratio - round(ratio)) > 1e-6:
    raise ValueError(
        f'{num_name} ({numerator}) must be an exact multiple of {den_name} '
        f'({denominator}), got a ratio of {ratio}.'
    )
  return round(ratio)


def run_integration_chunked(
    equations,
    coords: 'dinosaur.coordinate_systems.CoordinateSystem',
    physics_specs: 'dinosaur.units.SimUnitsProtocol',
    initial_state,
    ref_temps: np.ndarray,
    output_dir: str,
    exp_name: str,
    dt_si: 'units.Quantity' = 10 * units.minute,
    save_every: 'units.Quantity' = 1 * units.day,
    checkpoint_every: 'units.Quantity' = 50 * units.day,
    total_time: 'units.Quantity' = 1000 * units.day,
    filter_tau: float = 0.0087504,
    filter_order: float = 1.5,
    filter_cutoff: float = 0.8,
    resume: bool = True,
):
  """Runs a long integration in checkpointed chunks.

  Unlike `run_integration`, which accumulates the entire trajectory in
  memory for the whole run and only touches disk (if at all) at the very
  end, this never holds more than one chunk's worth of trajectory in
  memory, and is safe to kill and resume: with `resume=True` (the default),
  a restart picks up from the last *completed* chunk's checkpoint rather
  than starting over -- provided the run configuration (resolution, dt,
  etc.) matches what's recorded in the checkpoint, which is checked
  explicitly rather than silently trusted.

  A process killed mid-chunk loses at most that one chunk's compute (i.e.
  up to `checkpoint_every` of simulated time); checkpoints and chunk output
  files are only written after a chunk fully completes.

  Output layout in `output_dir`:
    `{exp_name}_chunk_0000.nc`, `{exp_name}_chunk_0001.nc`, ... -- one
      netCDF file of diagnostics per completed chunk, sampled every
      `save_every`. Load them together with `load_chunked_output`.
    `{exp_name}_checkpoint.npz` -- raw model state + metadata for the most
      recently *completed* chunk; overwritten (atomically) after each one.

  Args:
    equations: single equation system, or list to compose (e.g.
      `[primitive_equations, forcing]`).
    coords, physics_specs, initial_state, ref_temps: as in `run_integration`.
      `initial_state` is only used for a fresh start; if a matching
      checkpoint exists and `resume=True`, the checkpointed state is used
      instead.
    output_dir: directory for chunk/checkpoint files; created if missing.
    exp_name: filename prefix for output files.
    dt_si: integration timestep.
    save_every: interval between saved diagnostic snapshots within a chunk.
    checkpoint_every: simulated-time length of one chunk -- how often
      progress is flushed to disk. Must be an exact multiple of `save_every`.
    total_time: total simulated duration. Must be an exact multiple of
      `checkpoint_every`.
    filter_tau, filter_order, filter_cutoff: spectral filter parameters.
    resume: if `True` (default) and a matching checkpoint exists, continue
      from it instead of starting over.

  Returns:
    The final model `State` after `total_time`.
  """
  if isinstance(equations, (list, tuple)):
    equations = dinosaur.time_integration.compose_equations(list(equations))

  os.makedirs(output_dir, exist_ok=True)
  checkpoint_path = os.path.join(output_dir, f'{exp_name}_checkpoint.npz')

  save_every_days = save_every.to(units.day).magnitude
  checkpoint_every_days = checkpoint_every.to(units.day).magnitude
  total_time_days = total_time.to(units.day).magnitude

  outer_steps_per_chunk = _exact_ratio(
      checkpoint_every_days, save_every_days, 'checkpoint_every', 'save_every'
  )
  n_chunks = _exact_ratio(
      total_time_days, checkpoint_every_days, 'total_time', 'checkpoint_every'
  )
  inner_steps = int(save_every / dt_si)

  # `total_time` is deliberately excluded from this compatibility config:
  # extending (or shrinking) the target duration of a resumed run is a
  # legitimate operational choice, not a sign the checkpoint doesn't match
  # -- everything else here changes the per-step physics/numerics or the
  # shape of the checkpointed state, and so must match exactly to resume.
  config = {
      'dt_si_seconds': dt_si.to(units.second).magnitude,
      'save_every_days': save_every_days,
      'checkpoint_every_days': checkpoint_every_days,
      'longitude_wavenumbers': coords.horizontal.longitude_wavenumbers,
      'total_wavenumbers': coords.horizontal.total_wavenumbers,
      'longitude_nodes': coords.horizontal.longitude_nodes,
      'latitude_nodes': coords.horizontal.latitude_nodes,
      'layers': coords.vertical.layers,
  }

  start_chunk = 0
  state = initial_state
  if os.path.exists(checkpoint_path):
    if resume:
      state, saved_meta = load_checkpoint(checkpoint_path)
      mismatched = {
          k: (saved_meta.get(k), v)
          for k, v in config.items()
          if saved_meta.get(k) != v
      }
      if mismatched:
        raise ValueError(
            f'Checkpoint at {checkpoint_path} does not match the requested '
            f'run configuration: {mismatched}. Use a different exp_name/'
            f'output_dir for this configuration, or resume=False to '
            f'overwrite it and start over.'
        )
      start_chunk = saved_meta['chunks_completed']
      if start_chunk >= n_chunks:
        print(f'Checkpoint already covers all {n_chunks} chunks; nothing to do.')
        return state
      print(
          f'Resuming from checkpoint: {start_chunk}/{n_chunks} chunks already '
          f'done ({start_chunk * checkpoint_every_days:.0f} of '
          f'{total_time_days:.0f} days).'
      )
    else:
      print(f'resume=False: overwriting existing checkpoint at {checkpoint_path}.')

  dt = physics_specs.nondimensionalize(dt_si)
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
  advance_chunk = jax.jit(
      dinosaur.time_integration.trajectory_from_step(
          step_fn, outer_steps=outer_steps_per_chunk, inner_steps=inner_steps
      )
  )

  print(f'Running on device(s): {jax.devices()}')
  overall_start = time.time()
  for chunk_index in range(start_chunk, n_chunks):
    chunk_start = time.time()
    state, trajectory = jax.block_until_ready(advance_chunk(state))
    chunk_elapsed = time.time() - chunk_start

    day_offset = chunk_index * checkpoint_every_days
    times_days = day_offset + save_every_days * np.arange(
        1, outer_steps_per_chunk + 1
    )
    ds = trajectory_to_xarray(
        coords,
        physics_specs,
        jax.device_get(trajectory),
        times_days * units.day,
        ref_temps,
    )
    chunk_path = os.path.join(output_dir, f'{exp_name}_chunk_{chunk_index:04d}.nc')
    ds.to_netcdf(chunk_path)

    meta = dict(config)
    meta['chunks_completed'] = chunk_index + 1
    save_checkpoint(checkpoint_path, state, meta)

    total_elapsed = time.time() - overall_start
    chunks_done_this_run = chunk_index + 1 - start_chunk
    chunks_left = n_chunks - chunk_index - 1
    eta_seconds = (
        total_elapsed / chunks_done_this_run * chunks_left
        if chunks_done_this_run
        else float('nan')
    )
    print(
        f'chunk {chunk_index + 1}/{n_chunks} '
        f'({(chunk_index + 1) * checkpoint_every_days:.0f}/'
        f'{total_time_days:.0f} days) done in {chunk_elapsed:.1f}s, '
        f'wrote {chunk_path}; ETA {eta_seconds / 60:.1f} min',
        flush=True,
    )

  print(f'Finished all {n_chunks} chunks in {time.time() - overall_start:.1f}s.')
  return state


def load_chunked_output(output_dir: str, exp_name: str) -> xarray.Dataset:
  """Loads and concatenates all chunk files written by `run_integration_chunked`."""
  pattern = os.path.join(output_dir, f'{exp_name}_chunk_*.nc')
  paths = sorted(glob.glob(pattern))
  if not paths:
    raise FileNotFoundError(f'No chunk files found matching {pattern}')
  return xarray.open_mfdataset(paths, combine='by_coords')
