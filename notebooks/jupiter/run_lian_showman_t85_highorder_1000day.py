"""T85 Lian-Showman Jupiter run: 1000 days, original dt/tau, higher filter order.

Third in the stability-fix series, after halved-timestep (didn't help) and
10x-shorter-tau (made it worse -- full NaN by day ~900). This isolates
`order` alone: dt=5min and tau=0.0087504 are both back to the original
values, cutoff stays at 0.8, and only order changes, from 1.5 up to 18
(dinosaur's own library default for exponential_step_filter).

order=1.5 ramps damping up gently across the whole top 20% of the spectrum
(cutoff=0.8 to k=1). order=18 is a much sharper, more concentrated cutoff --
negligible damping until very close to k=1, then a steep wall. If the
problem is under-damped small-scale noise, this should target it more
precisely without over-damping legitimately-resolved mid-scale motion the
way the shorter-tau run did.

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
FILTER_TAU = 0.0087504  # unchanged from the original T85 run
FILTER_ORDER = 18  # up from 1.5 -- dinosaur's own library default
FILTER_CUTOFF = 0.8  # unchanged
SAVE_EVERY = 10 * units.day
CHECKPOINT_EVERY = 10 * units.day
TOTAL_TIME = 1000 * units.day

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'runs',
    'lian_showman_t85_highorder_1000day',
)
EXP_NAME = 'lian_showman_t85_highorder_1000day'


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
      filter_tau=FILTER_TAU,
      filter_order=FILTER_ORDER,
      filter_cutoff=FILTER_CUTOFF,
      resume=True,
  )


if __name__ == '__main__':
  main()
