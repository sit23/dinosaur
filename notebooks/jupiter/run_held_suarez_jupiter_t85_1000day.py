"""T85 standard Held-Suarez forcing, with Jupiter's radius/rotation/gravity.

Isolation experiment: identical setup (grid, dt, filter, duration) to
run_lian_showman_t85_1000day.py -- the reference case that developed a
pole-concentrated kinetic-energy blow-up after day ~700 -- but swapping in
the *canonical* (Earth-parameterized) Held-Suarez forcing instead of the
Lian-Showman one. Only `p0` is overridden (to 25 bar, so the forcing's
"surface" sigma=1 lines up with this model's actual deep reference pressure
rather than Earth's ~1e5 Pa); kf/ka/ks/minT/maxT/dTy/dThz/sigma_b are left
at their standard defaults on purpose.

If this ALSO blows up near the poles, that points at dinosaur's dynamical
core / Jupiter's fast rotation and large radius as the root cause, largely
independent of the Lian-Showman forcing's specific structure. If it stays
stable, that points at something specific to the Lian-Showman forcing
itself (e.g. its very different equilibrium-temperature gradients/taper
structure) as the more likely culprit.

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
P0 = 25e5 * units.pascal  # overridden from HeldSuarezForcingSigma's 1e5 Pa default
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
    'held_suarez_jupiter_t85_1000day',
)
EXP_NAME = 'held_suarez_jupiter_t85_1000day'


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
      'held_suarez', coords, physics_specs, ref_temps, orography, P0
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
