"""T85 Lian-Showman Jupiter run: 1000 days, del^4 hyperdiffusion instead of
the exponential spectral filter.

Same T85 setup as run_lian_showman_t85_1000day.py (the reference case that
developed a pole-concentrated blow-up after day ~700), but replacing
`exponential_step_filter` entirely with `horizontal_diffusion_step_filter`
(order=2, i.e. del^4/biharmonic) -- the same *style* of horizontal damping
Isca's spectral dynamical core actually relies on as its sole mechanism
(no separate cutoff-based filter on top). tau=10 days at the truncation
wavenumber matches Isca's own default `damping_coeff = 1/(10 days)`.

Unlike the exponential filter, this has no cutoff -- it damps every
wavenumber (very lightly at large scales, growing as a power law to the
full strength at the truncation limit), which the earlier code
investigation flagged as a real structural difference from what we'd been
using.

Safe to kill and re-launch identically -- resumes from the last checkpoint.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dinosaur
import jax
import jupiter_gcm_utils as jgu

units = dinosaur.scales.units

MAX_WAVENUMBER = 85
GAUSSIAN_NODES = 64
LAYERS = 60
SCALE_HEIGHTS = 11
P0 = 25e5 * units.pascal
DT = 5 * units.minute  # unchanged from the original T85 run
FILTER_TYPE = 'horizontal_diffusion'
FILTER_ORDER = 2  # del^4 / biharmonic, matching Isca's default damping_order
SAVE_EVERY = 10 * units.day
CHECKPOINT_EVERY = 10 * units.day
TOTAL_TIME = 1000 * units.day

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'runs',
    'lian_showman_t85_hdiff_1000day',
)
EXP_NAME = 'lian_showman_t85_hdiff_1000day'


def main():
  print(f'jax devices: {jax.devices()}', flush=True)

  horizontal = dinosaur.spherical_harmonic.Grid.construct(
      max_wavenumber=MAX_WAVENUMBER, gaussian_nodes=GAUSSIAN_NODES
  )
  vertical = dinosaur.sigma_coordinates.SigmaCoordinates.equidistant_log(
      LAYERS, SCALE_HEIGHTS
  )
  coords = dinosaur.coordinate_systems.CoordinateSystem(
      horizontal=horizontal, vertical=vertical
  )
  physics_specs = dinosaur.primitive_equations.PrimitiveEquationsSpecs.from_si()

  # horizontal_diffusion_step_filter's tau is a nondimensional-time
  # e-folding scale, same convention as exponential_step_filter's -- compute
  # it from the physical 10-day target the same way dt_si is nondimensionalized.
  filter_tau_nondim = physics_specs.nondimensionalize(10 * units.day)

  state, ref_temps, orography = jgu.initial_state(coords, physics_specs, p0=P0)
  equations = jgu.build_model_equations(
      'lian_showman', coords, physics_specs, ref_temps, orography, P0
  )

  jgu.run_integration_chunked(
      equations,
      coords,
      physics_specs,
      state,
      ref_temps,
      output_dir=OUTPUT_DIR,
      exp_name=EXP_NAME,
      dt_si=DT,
      save_every=SAVE_EVERY,
      checkpoint_every=CHECKPOINT_EVERY,
      total_time=TOTAL_TIME,
      filter_type=FILTER_TYPE,
      filter_tau=filter_tau_nondim,
      filter_order=FILTER_ORDER,
      resume=True,
  )


if __name__ == '__main__':
  main()
