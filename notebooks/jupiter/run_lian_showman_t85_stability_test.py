"""T85 Lian-Showman long-duration stability diagnostic.

Investigates the day ~200-250 blow-up seen in the T213/1000-day overnight
run (2026-08-20): same dt-halving CFL convention and same (unscaled, T42-
inherited) spectral filter parameters, but at a much cheaper resolution and
with a finer 10-day output/checkpoint cadence, so we can actually see when
(if at all) an instability develops rather than just "step 50 was fine,
step 250 was 100% NaN".

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
DT = 5 * units.minute  # same CFL-scaling convention as the T213/T170 runs
SAVE_EVERY = 10 * units.day
CHECKPOINT_EVERY = 10 * units.day
TOTAL_TIME = 300 * units.day

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'runs',
    'lian_showman_t85_stability_test',
)
EXP_NAME = 'lian_showman_t85_stability_test'


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
      resume=True,
  )


if __name__ == '__main__':
  main()
