"""T213 Lian-Showman Jupiter run: 1000 simulated days, with the SPEEDY/
jax-gcm-style per-field hyperdiffusion filter that fixed the T85 pole
instability (see run_lian_showman_t85_perfield_filter_1000day.py and
results_log.ipynb section 3.5).

T213 was the fastest-crashing configuration under the old uniform filter
(lian_showman_t213_1000day.py -- NaN around day 200-250), so this is the
key validation that the fix holds at higher resolution, not just T85,
before treating the fix as generally applicable.

Same resolution (mw=213/gn=160) and timestep (2 min) as the original T213
run, for a like-for-like comparison; only the filter differs.

Launched detached (nohup) so it survives independently of any particular
interactive session. Safe to kill and re-launch identically -- resumes
from the last completed 50-day checkpoint rather than starting over.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dinosaur
import jax
import jupiter_gcm_utils as jgu

units = dinosaur.scales.units

MAX_WAVENUMBER = 213
GAUSSIAN_NODES = 160
LAYERS = 60
SCALE_HEIGHTS = 11
P0 = 25e5 * units.pascal
DT = 2 * units.minute
SAVE_EVERY = 50 * units.day
CHECKPOINT_EVERY = 50 * units.day
TOTAL_TIME = 1000 * units.day

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'runs',
    'lian_showman_t213_perfield_filter_1000day',
)
EXP_NAME = 'lian_showman_t213_perfield_filter_1000day'


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
      filter_type='per_field_diffusion',
      resume=True,
  )


if __name__ == '__main__':
  main()
