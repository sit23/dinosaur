# Copyright 2023 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dry convective adjustment, for use alongside `grey_radiation`.

Ports the non-local dry convective adjustment scheme used by Isca's
`dry_convection_mod` (as configured for the `giant_planet` test case), which
grey-radiation giant-planet setups pair with radiation because a purely
radiative deep atmosphere is convectively unstable.

The algorithm, following Isca (`src/atmos_param/dry_convection/
dry_convection.f90`):

  1. Build a "parcel" target profile `tp` by an upward recursion from the
     bottom of the column, blending an isothermal profile (`gamma=0`) and a
     dry adiabat / constant potential temperature (`gamma=1`) via the Exner
     function:

       tp[bottom] = T[bottom]
       tp[k] = tp[k+1] * (1 + gamma * ((p[k] / p[k+1]) ** kappa - 1))

     `gamma` is dimensionless (a blend fraction, not a geometric lapse rate
     in K/km, despite how it's sometimes labeled) and `kappa` is the usual
     `R / cp`.

  2. A layer is "convecting" if it is buoyant relative to the parcel profile
     (`tp >= T`) *and* every layer below it, down to the bottom boundary, is
     too — i.e. the largest contiguous unstable region attached to the
     bottom of the column. This is a simplification of Isca's full CAPE/CIN
     search (which can also detect a second, disconnected unstable layer
     capped by a stable one below it); for a deep giant-planet atmosphere
     that is unstable in pure radiative equilibrium essentially down to its
     base, the bottom-attached region is the physically relevant one.

  3. Within the convecting region, the parcel profile is offset by a single
     constant so that it exactly conserves the pressure-weighted (i.e.
     mass-weighted, and hence energy-weighted, since `cp` is uniform)
     column integral of temperature relative to the original profile.

  4. The adjustment is applied as a Newtonian relaxation toward this
     energy-conserving target on timescale `tau`, not instantaneously; since
     step 3 already makes the *target* conserve column energy, the
     resulting tendency conserves it on every call, independent of `tau`.
"""

from dinosaur import coordinate_systems
from dinosaur import grey_radiation
from dinosaur import primitive_equations
from dinosaur import scales
from dinosaur import time_integration
from dinosaur import typing
from dinosaur import units as units_lib

import jax
import jax.numpy as jnp
import numpy as np

Quantity = typing.Quantity


def parcel_temperature(
    temperature: jnp.ndarray, p_full: jnp.ndarray, gamma: float, kappa: float
) -> jnp.ndarray:
  """Builds the parcel/target profile `tp` by upward recursion from the bottom.

  Args:
    temperature: ambient temperature at layer centers, shape `[n, ...]`, index
      0 at the top of the domain and index `n - 1` at the bottom.
    p_full: pressure at layer centers, same shape as `temperature`.
    gamma: dimensionless blend between isothermal (0) and dry-adiabatic (1)
      parcel lifting.
    kappa: `R / cp`.

  Returns:
    The parcel profile `tp`, same shape as `temperature`, with `tp[-1] ==
    temperature[-1]`.
  """
  exner_ratio = (p_full[:-1] / p_full[1:]) ** kappa
  a = 1 + gamma * (exner_ratio - 1)
  tp_rest = grey_radiation._affine_scan(
      a, jnp.zeros_like(a), temperature[-1], reverse=True
  )
  return jnp.concatenate([tp_rest, temperature[-1:]], axis=0)


def convecting_region(
    parcel_temp: jnp.ndarray, temperature: jnp.ndarray
) -> jnp.ndarray:
  """Boolean mask of the largest bottom-attached buoyant (unstable) region."""
  unstable = (parcel_temp >= temperature).astype(jnp.int32)
  return jax.lax.cummin(unstable, axis=0, reverse=True).astype(bool)


def adjustment_tendency(
    temperature: jnp.ndarray,
    parcel_temp: jnp.ndarray,
    dp_half: jnp.ndarray,
    tau: float,
) -> jnp.ndarray:
  """Temperature tendency from relaxing towards an energy-conserving target.

  Args:
    temperature: ambient temperature at layer centers, shape `[n, ...]`.
    parcel_temp: parcel/target profile from `parcel_temperature`.
    dp_half: pressure thickness of each layer, same shape as `temperature`.
    tau: relaxation timescale.

  Returns:
    Temperature tendency, same shape as `temperature`; exactly zero outside
    the convecting region, and mass-weighted-sum-zero within it.
  """
  in_region = convecting_region(parcel_temp, temperature)
  weight = jnp.where(in_region, dp_half, 0.0)
  offset = jnp.sum(weight * (temperature - parcel_temp), axis=0, keepdims=True) / (
      jnp.sum(weight, axis=0, keepdims=True)
  )
  target = jnp.where(in_region, parcel_temp + offset, temperature)
  return (target - temperature) / tau


class DryConvectiveAdjustment(time_integration.ExplicitODE):
  """Dry convective adjustment on sigma levels. See module docstring."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      physics_specs: units_lib.SimUnitsProtocol,
      reference_temperature: typing.Array,
      tau: Quantity = 21600 * scales.units.s,  # pyrefly: ignore[unsupported-operation]
      gamma: float = 1.0,
  ):
    """Initialize DryConvectiveAdjustment.

    Args:
      coords: horizontal and vertical descritization.
      physics_specs: object holding physical constants and definition of
        custom units to use for initialization of the state.
      reference_temperature: horizontal reference temperature at all
        altitudes.
      tau: relaxation timescale toward the convectively-adjusted profile.
      gamma: dimensionless blend between isothermal (0) and dry-adiabatic (1)
        parcel lifting; 1.0 (Isca's giant-planet default) targets a dry
        adiabat.
    """
    self.coords = coords
    self.physics_specs = physics_specs
    self.reference_temperature = reference_temperature
    self.tau = physics_specs.nondimensionalize(tau)
    self.gamma = gamma
    self.sigma = self.coords.vertical.centers[:, np.newaxis, np.newaxis]
    self.boundaries = self.coords.vertical.boundaries[:, np.newaxis, np.newaxis]

  def explicit_terms(
      self, state: primitive_equations.State
  ) -> primitive_equations.State:
    """Computes explicit temperature tendencies due to dry convection."""
    nodal_temperature_variation = self.coords.horizontal.to_nodal(
        state.temperature_variation
    )
    temperature = (
        self.reference_temperature[:, np.newaxis, np.newaxis]
        + nodal_temperature_variation
    )
    nodal_log_surface_pressure = self.coords.horizontal.to_nodal(
        state.log_surface_pressure
    )
    nodal_surface_pressure = jnp.exp(nodal_log_surface_pressure)
    p_full = self.sigma * nodal_surface_pressure
    p_half = self.boundaries * nodal_surface_pressure
    dp_half = p_half[1:] - p_half[:-1]

    tp = parcel_temperature(
        temperature, p_full, self.gamma, self.physics_specs.kappa
    )
    nodal_temperature_tendency = adjustment_tendency(
        temperature, tp, dp_half, self.tau
    )

    temperature_tendency = self.coords.horizontal.to_modal(
        nodal_temperature_tendency
    )

    return primitive_equations.State(
        vorticity=jnp.zeros_like(state.vorticity),  # pyrefly: ignore[unexpected-keyword]
        divergence=jnp.zeros_like(state.divergence),  # pyrefly: ignore[unexpected-keyword]
        temperature_variation=temperature_tendency,  # pyrefly: ignore[unexpected-keyword]
        log_surface_pressure=jnp.zeros_like(state.log_surface_pressure),  # pyrefly: ignore[unexpected-keyword]
    )
