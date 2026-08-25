"""T85 standard Held-Suarez forcing at Jupiter's rotation/size, weaker dTy.

Follow-up to run_held_suarez_jupiter_t85_1000day.py, which blew up fast
(day ~190) via a broadband, mid-latitude instability -- plausibly because
Held-Suarez's default 60K equator-to-pole gradient, tuned for Earth's
rotation rate, drives an unrealistically strong/baroclinically-unstable
jet once combined with Jupiter's ~2.4x faster rotation and ~11x larger
radius, before the flow ever gets anywhere near quasi-equilibrium.

This repeats that experiment with dTy lowered from 60K to 8K -- matching
LianShowmanForcing's own default, so this becomes a genuinely
apples-to-apples comparison against the Lian-Showman run: same weak
equator-to-pole forcing amplitude, canonical Held-Suarez everything else
(ka/ks/kf timescales, minT/maxT, dThz, sigma_b), only p0 and dTy changed
from the library defaults.

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
DTY = 8 * units.degK  # down from the 60K default, matching LianShowmanForcing's own default
DT = 5 * units.minute
FILTER_TAU = 0.0087504
FILTER_ORDER = 1.5
FILTER_CUTOFF = 0.8
SAVE_EVERY = 10 * units.day
CHECKPOINT_EVERY = 10 * units.day
TOTAL_TIME = 1000 * units.day

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'runs',
    'held_suarez_jupiter_lowdty_t85_1000day',
)
EXP_NAME = 'held_suarez_jupiter_lowdty_t85_1000day'


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
      'held_suarez',
      coords,
      physics_specs,
      ref_temps,
      orography,
      P0,
      forcing_kwargs={'dTy': DTY},
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
      filter_tau=FILTER_TAU,
      filter_order=FILTER_ORDER,
      filter_cutoff=FILTER_CUTOFF,
      resume=True,
  )


if __name__ == '__main__':
  main()
