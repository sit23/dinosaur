"""T85 Lian-Showman Jupiter run: 1000 days, SPEEDY/ECHAM/jax-gcm-style
per-field hyperdiffusion (independent divergence/vorticity/temperature
filters, divergence damped much harder) instead of one filter applied
uniformly to the whole state.

Every previous filter variant tried against this reference case (halved
timestep, shorter tau, higher order, uniform del^4 hyperdiffusion) made the
T85 pole instability blow up *faster*, not slower -- see
run_lian_showman_t85_1000day.py (the reference, KE still rising but finite
through day 1000) and its siblings (_halfdt_, _shorttau_, _highorder_,
_hdiff_). Investigation into jax-gcm (github.com/climate-analytics-lab/
jax-gcm, which wraps dinosaur as its dycore) found it applies three
*independent* hyperdiffusion filters -- divergence, vorticity+humidity,
temperature -- rather than one uniform filter, with divergence damped much
harder (SPEEDY defaults: 2h/del^2 for divergence vs 12h/del^4 vorticity,
24h/del^4 temperature). Divergence is the field most directly coupled to
fast gravity-wave/numerical noise -- and the pole problem we diagnosed
(eddy KE concentrated at the highest latitudes, dominated by m=1,2) is
exactly the kind of thing that mechanism would target -- so this is the
first attempt at fixing the actual pole instability rather than
characterizing it further.

Same dt (5 min), resolution (T85/60 levels), and output/checkpoint cadence
as the reference case, for a like-for-like comparison. Safe to kill and
re-launch identically -- resumes from the last checkpoint.
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
DT = 5 * units.minute
SAVE_EVERY = 10 * units.day
CHECKPOINT_EVERY = 10 * units.day
TOTAL_TIME = 1000 * units.day

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'runs',
    'lian_showman_t85_perfield_filter_1000day',
)
EXP_NAME = 'lian_showman_t85_perfield_filter_1000day'


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
      # SPEEDY defaults (2h/12h/24h at del^2/del^4/del^4) -- an Earth-tuned
      # starting point, not re-derived for Jupiter's rotation rate.
      resume=True,
  )


if __name__ == '__main__':
  main()
