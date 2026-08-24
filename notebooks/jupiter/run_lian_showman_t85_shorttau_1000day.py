"""T85 Lian-Showman Jupiter run: 1000 days, original dt, much shorter filter tau.

Companion to run_lian_showman_t85_halfdt_1000day.py: that one changes the
timestep and holds the filter fixed; this one holds the *original* T85
timestep (5 min, the one that showed the polar instability) fixed and
instead makes the spectral filter's damping much stronger, to isolate
whether damping alone (no CFL/timestep change) fixes the blow-up.

tau is reduced 10x from 0.0087504 to 0.00087504 (nondimensional) --
physical e-folding time at the top wavenumber (k=1) drops from ~24.9s to
~2.5s. order (1.5) and cutoff (0.8) are unchanged, so this only changes how
*strongly* the same fraction of the spectrum (top 20%, k>0.8) gets damped,
not which modes are touched.

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
FILTER_TAU = 0.00087504  # 10x shorter than the 0.0087504 used everywhere else
FILTER_ORDER = 1.5
FILTER_CUTOFF = 0.8
SAVE_EVERY = 10 * units.day
CHECKPOINT_EVERY = 10 * units.day
TOTAL_TIME = 1000 * units.day

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'runs',
    'lian_showman_t85_shorttau_1000day',
)
EXP_NAME = 'lian_showman_t85_shorttau_1000day'


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
