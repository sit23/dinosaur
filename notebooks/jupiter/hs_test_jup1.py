import functools
import jax
import dinosaur
import numpy as np
import matplotlib.pyplot as plt
import xarray
import pdb
import time
import os

units = dinosaur.scales.units

def dimensionalize(x: xarray.DataArray, unit: units.Unit) -> xarray.DataArray:
  """Dimensionalizes `xarray.DataArray`s."""
  dimensionalize = functools.partial(physics_specs.dimensionalize, unit=unit)
  return xarray.apply_ufunc(dimensionalize, x)


# Formatting trajectory to xarray.Dataset
def trajectory_to_xarray(coords, trajectory, times):

  trajectory_dict, _ = dinosaur.pytree_utils.as_dict(trajectory)
  u, v = dinosaur.spherical_harmonic.vor_div_to_uv_nodal(
      coords.horizontal, trajectory.vorticity, trajectory.divergence)

  u_dim = dimensionalize(u, units.meter/units.second)
  v_dim = dimensionalize(v, units.meter/units.second)
  vort_dim = dimensionalize(trajectory.vorticity, 1./units.second)
  div_dim  = dimensionalize(trajectory.divergence, 1./units.second)


  trajectory_dict.update({'u': u_dim, 'v': v_dim, 'vorticity':vort_dim, 'divergence':div_dim})
  nodal_trajectory_fields = dinosaur.coordinate_systems.maybe_to_nodal(
      trajectory_dict, coords=coords)
  trajectory_ds = dinosaur.xarray_utils.data_to_xarray(
      nodal_trajectory_fields, coords=coords, times=times)

  trajectory_ds['surface_pressure'] = dimensionalize(np.exp(trajectory_ds.log_surface_pressure[:, 0, :,:]), units.pascal)

  temperature = dinosaur.xarray_utils.temperature_variation_to_absolute(
      trajectory_ds.temperature_variation.data, ref_temps)
  trajectory_ds = trajectory_ds.assign(
      temperature=(trajectory_ds.temperature_variation.dims, dimensionalize(temperature, units.degK)))

  total_layer_ke = coords.horizontal.integrate(u**2 + v**2)
  total_ke_cumulative = dinosaur.sigma_coordinates.cumulative_sigma_integral(
      total_layer_ke, coords.vertical, axis=-1)
  total_ke = total_ke_cumulative[..., -1]
  trajectory_ds = trajectory_ds.assign(total_kinetic_energy=(('time'), dimensionalize(total_ke, units.meter**2/units.second**2)))
  return trajectory_ds

def ds_held_suarez_forcing(coords, hs):
  grid = coords.horizontal
  sigma = coords.vertical.centers
  lon, _ = grid.nodal_mesh
  surface_pressure = physics_specs.nondimensionalize(p0) * np.ones_like(lon)
  dims = ('sigma', 'lon', 'lat')
  return xarray.Dataset(
      data_vars={
          'surface_pressure': (('lon', 'lat'), surface_pressure),
          'eq_temp': (dims, hs.equilibrium_temperature(surface_pressure)),
          'kt': (dims, hs.kt()),
          'kv': ('sigma', hs.kv()[:, 0, 0]),
      },
      coords={'lon': grid.nodal_axes[0] * 180 / np.pi,
              'lat': np.arcsin(grid.nodal_axes[1]) * 180 / np.pi,
              'sigma': sigma},
  )

def linspace_step(start, stop, step):
  num = round((stop - start) / step) + 1
  return np.linspace(start, stop, num)


#To change this to Jupiter we need to:

# * Find the rotation rate and radius specification - this seems to be in scales.py, then everything is non-dimensionalised for the actual calculations.

print('welcome to dinosaur')
# Resolution

units = dinosaur.scales.units
physics_specs = dinosaur.primitive_equations.PrimitiveEquationsSpecs.from_si()

#set simulation parameters
exp_name = 'lian_showman_v1_1_day_short'
output_file_name = f'{exp_name}.nc'
overwrite_existing_data = False

if os.path.isfile(output_file_name): 
  if overwrite_existing_data:
    raise FileExistsError('Output file already exists and overwriting is disabled - stopping')
  else:
    raise RuntimeWarning('Output file already exists and overwriting is ENABLED - continuing')
  
layers = 60
coords = dinosaur.coordinate_systems.CoordinateSystem(
    horizontal=dinosaur.spherical_harmonic.Grid.T42(),
    vertical=dinosaur.sigma_coordinates.SigmaCoordinates.equidistant_log(layers, 11))

#set physical properties of system
p0 = 25*1e5 * units.pascal


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
save_every = 1 * units.day #output frequency
total_time = 1 * units.day #total time for simulation

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
hs_forcing = dinosaur.held_suarez.LianShowmanForcing(
    coords=coords,
    physics_specs=physics_specs,
    reference_temperature=ref_temps,
    p0=p0)

forcing_ds = ds_held_suarez_forcing(coords, hs_forcing)

# Friction coefficient {form-width: "40%"}
kv = forcing_ds['kv']
kv_si = dimensionalize(kv, 1 / units.day)
kv_si.plot(size=5)

# Thermal relaxation coefficient {form-width: "40%"}
kt_array = forcing_ds['kt']
levels = 30
kt_array_si = dimensionalize(kt_array, 1 / units.day)
p = kt_array_si.isel(lon=0).plot.contour(x='lat', y='sigma', levels=levels,
                                         size=5, aspect=1.5)
ax = plt.gca()
ax.set_ylim((1, 0))
ax.set_title('Kt at lon = 0')
plt.colorbar(p);

# Radiative equilibrium temperature {form-width: "40%"}
teq_array = forcing_ds['eq_temp']
teq_array_si = dimensionalize(teq_array, units.degK)
levels = 30
p = teq_array_si.isel(lon=0).plot.contour(x='lat', y='sigma', levels=levels,
                                          size=5, aspect=1.5)
ax = plt.gca()
ax.set_ylim((1, 0));
#plt.colorbar(p);

plt.figure()
teq_array_si.isel(lon=0).sel(lat=0., method='nearest').plot.line(y='sigma', yincrease=False)
ax = plt.gca()
ax.set_yscale('log')

# Radiative equilibrium potential temperature {form-width: "40%"}
temperature = dimensionalize(forcing_ds['eq_temp'], units.degK)
surface_pressure = dimensionalize(forcing_ds['surface_pressure'], units.pascal)
pressure = forcing_ds.sigma * surface_pressure
kappa = dinosaur.scales.KAPPA  # kappa = R / cp, R = gas constant, cp = specific heat capacity
potential_temperature = temperature * (pressure / p0)**-kappa

plt.figure()
potential_temperature.isel(lon=0).sel(lat=0., method='nearest').plot.line(y='sigma', yincrease=False)
ax = plt.gca()
ax.set_yscale('log')

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
ds.to_netcdf(f'{exp_name}.nc')