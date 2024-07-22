import functools
import jax
import dinosaur
import numpy as np
import matplotlib.pyplot as plt
import xarray
import pdb
import time

def dimensionalize(x: xarray.DataArray, unit: units.Unit) -> xarray.DataArray:
  """Dimensionalizes `xarray.DataArray`s."""
  dimensionalize = functools.partial(physics_specs.dimensionalize, unit=unit)
  return xarray.apply_ufunc(dimensionalize, x)


# Formatting trajectory to xarray.Dataset
def trajectory_to_xarray(coords, trajectory, times):

  trajectory_dict, _ = dinosaur.pytree_utils.as_dict(trajectory)
  u, v = dinosaur.spherical_harmonic.vor_div_to_uv_nodal(
      coords.horizontal, trajectory.vorticity, trajectory.divergence)
  trajectory_dict.update({'u': u, 'v': v})
  nodal_trajectory_fields = dinosaur.coordinate_systems.maybe_to_nodal(
      trajectory_dict, coords=coords)
  trajectory_ds = dinosaur.xarray_utils.data_to_xarray(
      nodal_trajectory_fields, coords=coords, times=times)

  trajectory_ds['surface_pressure'] = np.exp(trajectory_ds.log_surface_pressure[:, 0, :,:])
  temperature = dinosaur.xarray_utils.temperature_variation_to_absolute(
      trajectory_ds.temperature_variation.data, ref_temps)
  trajectory_ds = trajectory_ds.assign(
      temperature=(trajectory_ds.temperature_variation.dims, temperature))

  total_layer_ke = coords.horizontal.integrate(u**2 + v**2)
  total_ke_cumulative = dinosaur.sigma_coordinates.cumulative_sigma_integral(
      total_layer_ke, coords.vertical, axis=-1)
  total_ke = total_ke_cumulative[..., -1]
  trajectory_ds = trajectory_ds.assign(total_kinetic_energy=(('time'), total_ke))
  return trajectory_ds


print('welcome to dinosaur')
# Resolution

units = dinosaur.scales.units
physics_specs = dinosaur.primitive_equations.PrimitiveEquationsSpecs.from_si()

#set simulation parameters

layers = 24
coords = dinosaur.coordinate_systems.CoordinateSystem(
    horizontal=dinosaur.spherical_harmonic.Grid.T42(),
    vertical=dinosaur.sigma_coordinates.SigmaCoordinates.equidistant(layers))

#set physical properties of system
p0 = 100e3 * units.pascal

#magnitude of random surface pressure perturbation in initial conditions
p1 = 5e3 * units.pascal

#set some random perturbation for initial conditions
rng_key = jax.random.PRNGKey(0)


# Set initial state function using resolution, physics specs, p0 and p1

initial_state_fn, aux_features = dinosaur.primitive_equations_states.isothermal_rest_atmosphere(
    coords=coords,
    physics_specs=physics_specs,
    p0=p0,
    p1=p1)

# set initial state using random perturbation (rng_key) and isothermal rest state
initial_state = initial_state_fn(rng_key)

#This is a uniform reference temperature, defaulting to 288.
ref_temps = aux_features[dinosaur.xarray_utils.REF_TEMP_KEY]

#setting orography, which has in fact already been specifed in the isothermal rest atmosphere function, and sits within the aux_features.
orography = dinosaur.primitive_equations.truncated_modal_orography(
    aux_features[dinosaur.xarray_utils.OROGRAPHY], coords)


print('now about to do some integrating')
# Integration settings
dt_si = 10 * units.minute #timestep
save_every = 30 * units.day #output frequency
total_time = 30 * units.day #total time for simulation

inner_steps = int(save_every / dt_si) #number of steps for output
outer_steps = int(total_time / save_every) #number of steps for entire simulation
dt = physics_specs.nondimensionalize(dt_si) #non-dimensionalise dt

# Governing equations
primitive = dinosaur.primitive_equations.PrimitiveEquations(
    ref_temps,
    orography,
    coords,
    physics_specs)

#Here's where the actual forcing is defined
hs_forcing = dinosaur.held_suarez.HeldSuarezForcing(
    coords=coords,
    physics_specs=physics_specs,
    reference_temperature=ref_temps,
    p0=p0)

#Here's where we tell the time tintegration what equation sets to use, i.e. here a combination of the primitive equations and the hs_forcing fields
primitive_with_hs = dinosaur.time_integration.compose_equations([primitive, hs_forcing])

#specify the timesteepper, the function to step (primitive_with_hs) and the timestep dt (non-dimensionalised)

step_fn = dinosaur.time_integration.imex_rk_sil3(primitive_with_hs, dt)

#looks like a small scale filter
filters = [
    dinosaur.time_integration.exponential_step_filter(
        coords.horizontal, dt, tau=0.0087504, order=1.5, cutoff=0.8),
]
#setting function to step with a filter applied
step_fn = dinosaur.time_integration.step_with_filters(step_fn, filters)

# Finally we get to the integration function, which is the step_fn and the number of outer and inner steps
integrate_fn = jax.jit(dinosaur.time_integration.trajectory_from_step(
    step_fn,
    outer_steps=outer_steps,
    inner_steps=inner_steps))

# Define trajectory times, expects start_with_input=False
times = save_every * np.arange(1, outer_steps+1)

#now actually do some integrating
print('ready to integrate')
start_time = time.time()
final, trajectory = jax.block_until_ready(integrate_fn(initial_state))
end_time = time.time()
print(end_time-start_time, ' seconds have passed')
print('finished integrating')

#turn the output into an xarray dataset
ds = trajectory_to_xarray(coords, jax.device_get(trajectory), times)