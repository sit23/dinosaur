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

"""Code for generating conditions for the Held-Suarez test case.

This test case is based on

  Held, I. M., and M. J. Suarez, 1994: "A proposal for the intercomparison of
  the dynamical cores of atmospheric general circulation models."
  Bulletin of the American Meteorological Society, 75, 1825–1830.
"""

import dataclasses
from dinosaur import coordinate_systems
from dinosaur import hybrid_coordinates
from dinosaur import primitive_equations
from dinosaur import scales
from dinosaur import time_integration
from dinosaur import typing
from dinosaur import units
import jax
import jax.numpy as jnp
import numpy as np


Quantity = typing.Quantity

# Variable names used to match format in Held-Suarez paper.
# pylint: disable=invalid-name


# TODO(dkochkov): Consider passing Grid and levels separately to enable better
# pytype checking.


class HeldSuarezForcingSigma(time_integration.ExplicitODE):
  """The Held-Suarez forcing specification on sigma levels."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      physics_specs: units.SimUnitsProtocol,
      reference_temperature: typing.Array,
      p0: Quantity = 1e5 * scales.units.pascal,  # pyrefly: ignore[unsupported-operation]
      sigma_b: Quantity = 0.7,
      kf: Quantity = 1 / (1 * scales.units.day),  # pyrefly: ignore[unsupported-operation]
      ka: Quantity = 1 / (40 * scales.units.day),  # pyrefly: ignore[unsupported-operation]
      ks: Quantity = 1 / (4 * scales.units.day),  # pyrefly: ignore[unsupported-operation]
      minT: Quantity = 200 * scales.units.degK,  # pyrefly: ignore[unsupported-operation]
      maxT: Quantity = 315 * scales.units.degK,  # pyrefly: ignore[unsupported-operation]
      dTy: Quantity = 60 * scales.units.degK,  # pyrefly: ignore[unsupported-operation]
      dThz: Quantity = 10 * scales.units.degK,  # pyrefly: ignore[unsupported-operation]
  ):
    """Initialize HeldSuarezForcingSigma.

    Args:
      coords: horizontal and vertical descritization.
      physics_specs: object holding physical constants and definition of custom
        units to use for initialization of the state.
      reference_temperature: horizontal reference temperature at all altitudes.
      p0: reference surface pressure.
      sigma_b: sigma level of effective planetary boundary layer.
      kf: coefficient of friction for Rayleigh drag.
      ka: coefficient of thermal relaxation in upper atmosphere.
      ks: coefficient of thermal relaxation at earth surface on the equator.
      minT: lower temperature bound of radiative equilibrium.
      maxT: upper temperature bound of radiative equilibrium.
      dTy: horizontal temperature variation of radiative equilibrium.
      dThz: vertical temperature variation of radiative equilibrium.
    """
    self.coords = coords
    self.physics_specs = physics_specs
    self.reference_temperature = reference_temperature
    self.p0 = physics_specs.nondimensionalize(p0)
    self.sigma_b = sigma_b
    self.kf = physics_specs.nondimensionalize(kf)
    self.ka = physics_specs.nondimensionalize(ka)
    self.ks = physics_specs.nondimensionalize(ks)
    self.minT = physics_specs.nondimensionalize(minT)
    self.maxT = physics_specs.nondimensionalize(maxT)
    self.dTy = physics_specs.nondimensionalize(dTy)
    self.dThz = physics_specs.nondimensionalize(dThz)
    # Coordinates
    self.sigma = self.coords.vertical.centers
    _, sin_lat = self.coords.horizontal.nodal_mesh
    self.lat = np.arcsin(sin_lat)

  def kv(self):
    kv_coeff = self.kf * (
        np.maximum(0, (self.sigma - self.sigma_b) / (1 - self.sigma_b))
    )
    return kv_coeff[:, np.newaxis, np.newaxis]

  def kt(self):
    cutoff = np.maximum(0, (self.sigma - self.sigma_b) / (1 - self.sigma_b))
    return self.ka + (self.ks - self.ka) * (
        cutoff[:, np.newaxis, np.newaxis] * np.cos(self.lat) ** 4
    )

  def equilibrium_temperature(self, nodal_surface_pressure):
    p_over_p0 = (
        self.sigma[:, np.newaxis, np.newaxis] * nodal_surface_pressure / self.p0
    )
    temperature = p_over_p0**self.physics_specs.kappa * (
        self.maxT
        - self.dTy * np.sin(self.lat) ** 2
        - self.dThz * jnp.log(p_over_p0) * np.cos(self.lat) ** 2
    )
    return jnp.maximum(self.minT, temperature)

  def explicit_terms(
      self, state: primitive_equations.State
  ) -> primitive_equations.State:
    """Computes explicit tendencies due to Held-Suarez forcing."""
    aux_state = primitive_equations.compute_diagnostic_state_sigma(
        state=state, coords=self.coords
    )

    # Nodal velocity tendencies
    # here nodal velocity includes 1/cos(lat) factor that will be removed at
    # the end when we compute divergence and vorticity tendncies using
    # curl_cos_lat and div_cos_lat.
    nodal_velocity_tendency = jax.tree.map(
        lambda x: -self.kv() * x / self.coords.horizontal.cos_lat**2,
        aux_state.cos_lat_u,
    )

    # Nodal temperature tendency
    nodal_temperature = (
        self.reference_temperature[:, np.newaxis, np.newaxis]
        + aux_state.temperature_variation
    )
    nodal_log_surface_pressure = self.coords.horizontal.to_nodal(
        state.log_surface_pressure
    )
    nodal_surface_pressure = jnp.exp(nodal_log_surface_pressure)
    Teq = self.equilibrium_temperature(nodal_surface_pressure)
    nodal_temperature_tendency = -self.kt() * (nodal_temperature - Teq)

    # Convert to modal
    temperature_tendency = self.coords.horizontal.to_modal(
        nodal_temperature_tendency
    )
    velocity_tendency = self.coords.horizontal.to_modal(nodal_velocity_tendency)
    vorticity_tendency = self.coords.horizontal.curl_cos_lat(velocity_tendency)
    divergence_tendency = self.coords.horizontal.div_cos_lat(velocity_tendency)

    # Zero log_surface_pressure tendency
    log_surface_pressure_tendency = jnp.zeros_like(state.log_surface_pressure)

    return primitive_equations.State(
        vorticity=vorticity_tendency,  # pyrefly: ignore[unexpected-keyword]
        divergence=divergence_tendency,  # pyrefly: ignore[unexpected-keyword]
        temperature_variation=temperature_tendency,  # pyrefly: ignore[unexpected-keyword]
        log_surface_pressure=log_surface_pressure_tendency,  # pyrefly: ignore[unexpected-keyword]
    )


class LianShowmanForcing(time_integration.ExplicitODE):
  """Jupiter-like relaxation forcing on sigma levels.

  This is a Held-Suarez-style Newtonian relaxation forcing, but with an
  equilibrium temperature profile appropriate for a giant planet rather than
  Earth, following

    Lian, Y., and A. P. Showman, 2008: "Deep jets on gas-giant planets."
    Icarus, 194, 597-615.
    Lian, Y., and A. P. Showman, 2010: "Generation of equatorial jets by
    large-scale latent heating on the giant planets." Icarus, 207, 373-393.

  The equilibrium temperature is built from three pieces stitched together in
  log-pressure: an isothermal deep interior (`const_t`, for p <=
  `p_low_taper`), a background profile with constant potential temperature
  `const_theta` referenced to `p_crossover` (for p >= `p_high_taper`), and a
  cubic polynomial in log(p) connecting the two continuously (value and
  derivative) in between.
  """

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      physics_specs: units.SimUnitsProtocol,
      reference_temperature: typing.Array,
      p0: Quantity = 1e5 * scales.units.pascal,  # pyrefly: ignore[unsupported-operation]
      sigma_b: Quantity = 0.8,
      kf: Quantity = 0 / (1 * scales.units.day),  # pyrefly: ignore[unsupported-operation]
      ks: Quantity = 1 / (400 * scales.units.day),  # pyrefly: ignore[unsupported-operation]
      ka: Quantity = 1 / (40 * scales.units.day),  # pyrefly: ignore[unsupported-operation]
      dTy: Quantity = 8 * scales.units.degK,  # pyrefly: ignore[unsupported-operation]
      const_t: Quantity = 140.0 * scales.units.degK,  # pyrefly: ignore[unsupported-operation]
      const_theta: Quantity = 155.5 * scales.units.degK,  # pyrefly: ignore[unsupported-operation]
      p_crossover: Quantity = 0.95e5 * scales.units.pascal,  # pyrefly: ignore[unsupported-operation]
      p_low_taper: Quantity = 0.4e5 * scales.units.pascal,  # pyrefly: ignore[unsupported-operation]
      p_high_taper: Quantity = 1.5e5 * scales.units.pascal,  # pyrefly: ignore[unsupported-operation]
  ):
    """Initialize LianShowmanForcing.

    Args:
      coords: horizontal and vertical descritization.
      physics_specs: object holding physical constants and definition of
        custom units to use for initialization of the state.
      reference_temperature: horizontal reference temperature at all
        altitudes.
      p0: reference surface pressure.
      sigma_b: sigma level of effective planetary boundary layer.
      kf: coefficient of friction for Rayleigh drag.
      ks: coefficient of thermal relaxation near the top of the domain.
      ka: coefficient of thermal relaxation in the free atmosphere.
      dTy: horizontal (latitudinal) temperature perturbation added to the
        equilibrium temperature.
      const_t: isothermal deep-interior temperature.
      const_theta: potential temperature of the background profile above the
        tropopause, referenced to `p_crossover`.
      p_crossover: reference pressure for `const_theta`.
      p_low_taper: pressure below which the profile is exactly `const_t`.
      p_high_taper: pressure above which the profile follows `const_theta`.
    """
    self.coords = coords
    self.physics_specs = physics_specs
    self.reference_temperature = reference_temperature
    self.p0 = physics_specs.nondimensionalize(p0)
    self.sigma_b = sigma_b
    self.kf = physics_specs.nondimensionalize(kf)
    self.ka = physics_specs.nondimensionalize(ka)
    self.ks = physics_specs.nondimensionalize(ks)
    self.dTy = physics_specs.nondimensionalize(dTy)
    self.const_t = physics_specs.nondimensionalize(const_t)
    self.const_theta = physics_specs.nondimensionalize(const_theta)
    self.p_crossover = physics_specs.nondimensionalize(p_crossover)
    self.p_low_taper = physics_specs.nondimensionalize(p_low_taper)
    self.p_high_taper = physics_specs.nondimensionalize(p_high_taper)

    # Coordinates
    self.sigma = self.coords.vertical.centers
    _, sin_lat = self.coords.horizontal.nodal_mesh
    self.lat = np.arcsin(sin_lat)

    # Solve for the cubic polynomial (in log(p)) connecting the isothermal
    # deep interior to the constant-potential-temperature background profile
    # continuously in both value and derivative at `p_low_taper`/
    # `p_high_taper`.
    x1 = jnp.log(self.p_high_taper)
    x2 = jnp.log(self.p_low_taper)
    kappa = self.physics_specs.kappa
    D = jnp.log(self.const_t) + kappa * jnp.log(self.p_crossover)
    C = jnp.log(self.const_theta)

    matrix_for_cubic = jnp.array([
        [x1**3, x1**2, x1, 1.0],
        [3 * x1**2, 2 * x1, 1.0, 0.0],
        [x2**3, x2**2, x2, 1.0],
        [3 * x2**2, 2 * x2, 1.0, 0.0],
    ])
    rhs_of_matrix_equation = jnp.array(
        [C, 0.0, D - kappa * x2, -kappa]
    )
    self.coeffs = jnp.linalg.solve(matrix_for_cubic, rhs_of_matrix_equation)

  def kv(self):
    kv_coeff = self.kf * (
        np.maximum(0, (self.sigma - self.sigma_b) / (1 - self.sigma_b))
    )
    return kv_coeff[:, np.newaxis, np.newaxis]

  def kt(self):
    cutoff = np.maximum(0, (self.sigma - self.sigma_b) / (1 - self.sigma_b))
    return self.ka + (self.ks - self.ka) * cutoff[:, np.newaxis, np.newaxis]

  def equilibrium_temperature(self, nodal_surface_pressure):
    """Computes the Jupiter-like equilibrium temperature profile."""
    kappa = self.physics_specs.kappa
    p_values = self.sigma[:, np.newaxis, np.newaxis] * nodal_surface_pressure

    # Constant-potential-temperature background above the tropopause.
    temps_from_theta = (
        self.const_theta * (p_values / self.p_crossover) ** kappa
    )
    background_temp = jnp.where(
        p_values >= self.p_high_taper, temps_from_theta, 0.0
    )
    # Isothermal deep interior.
    background_temp += jnp.where(
        p_values <= self.p_low_taper, self.const_t, 0.0
    )

    # Cubic taper (in log(p)) connecting the two regimes.
    log_p = jnp.log(p_values)
    taper_potential_temp = jnp.exp(
        self.coeffs[0] * log_p**3
        + self.coeffs[1] * log_p**2
        + self.coeffs[2] * log_p
        + self.coeffs[3]
    )
    taper_temp = taper_potential_temp * (p_values / self.p_crossover) ** kappa
    in_taper_region = jnp.logical_and(
        p_values > self.p_low_taper, p_values < self.p_high_taper
    )
    background_temp += jnp.where(in_taper_region, taper_temp, 0.0)

    perturbation = self.dTy * np.cos(self.lat) ** 2
    return background_temp + perturbation

  def explicit_terms(
      self, state: primitive_equations.State
  ) -> primitive_equations.State:
    """Computes explicit tendencies due to Lian-Showman forcing."""
    aux_state = primitive_equations.compute_diagnostic_state_sigma(
        state=state, coords=self.coords
    )

    # Nodal velocity tendencies
    # here nodal velocity includes 1/cos(lat) factor that will be removed at
    # the end when we compute divergence and vorticity tendncies using
    # curl_cos_lat and div_cos_lat.
    nodal_velocity_tendency = jax.tree.map(
        lambda x: -self.kv() * x / self.coords.horizontal.cos_lat**2,
        aux_state.cos_lat_u,
    )

    # Nodal temperature tendency
    nodal_temperature = (
        self.reference_temperature[:, np.newaxis, np.newaxis]
        + aux_state.temperature_variation
    )
    nodal_log_surface_pressure = self.coords.horizontal.to_nodal(
        state.log_surface_pressure
    )
    nodal_surface_pressure = jnp.exp(nodal_log_surface_pressure)
    Teq = self.equilibrium_temperature(nodal_surface_pressure)
    nodal_temperature_tendency = -self.kt() * (nodal_temperature - Teq)

    # Convert to modal
    temperature_tendency = self.coords.horizontal.to_modal(
        nodal_temperature_tendency
    )
    velocity_tendency = self.coords.horizontal.to_modal(nodal_velocity_tendency)
    vorticity_tendency = self.coords.horizontal.curl_cos_lat(velocity_tendency)
    divergence_tendency = self.coords.horizontal.div_cos_lat(velocity_tendency)

    # Zero log_surface_pressure tendency
    log_surface_pressure_tendency = jnp.zeros_like(state.log_surface_pressure)

    return primitive_equations.State(
        vorticity=vorticity_tendency,  # pyrefly: ignore[unexpected-keyword]
        divergence=divergence_tendency,  # pyrefly: ignore[unexpected-keyword]
        temperature_variation=temperature_tendency,  # pyrefly: ignore[unexpected-keyword]
        log_surface_pressure=log_surface_pressure_tendency,  # pyrefly: ignore[unexpected-keyword]
    )


class HeldSuarezForcingHybrid(time_integration.ExplicitODE):
  """The Held-Suarez test problem forcing specification on hybrid levels."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      physics_specs: units.SimUnitsProtocol,
      reference_temperature: typing.Array,
      p0: Quantity = 1e5 * scales.units.pascal,  # pyrefly: ignore[unsupported-operation]
      sigma_b: float = 0.7,
      kf: Quantity = 1 / (1 * scales.units.day),  # pyrefly: ignore[unsupported-operation]
      ka: Quantity = 1 / (40 * scales.units.day),  # pyrefly: ignore[unsupported-operation]
      ks: Quantity = 1 / (4 * scales.units.day),  # pyrefly: ignore[unsupported-operation]
      minT: Quantity = 200 * scales.units.degK,  # pyrefly: ignore[unsupported-operation]
      maxT: Quantity = 315 * scales.units.degK,  # pyrefly: ignore[unsupported-operation]
      dTy: Quantity = 60 * scales.units.degK,  # pyrefly: ignore[unsupported-operation]
      dThz: Quantity = 10 * scales.units.degK,  # pyrefly: ignore[unsupported-operation]
      hpa_quantity: Quantity = scales.units.hPa,
  ):
    """Initializes HybridHeldSuarezForcingHybrid."""
    self.coords = coords
    self.physics_specs = physics_specs
    self.reference_temperature = reference_temperature
    self.p0 = self.physics_specs.nondimensionalize(p0)
    self.sigma_b = sigma_b
    self.kf = self.physics_specs.nondimensionalize(kf)
    self.ka = self.physics_specs.nondimensionalize(ka)
    self.ks = self.physics_specs.nondimensionalize(ks)
    self.minT = self.physics_specs.nondimensionalize(minT)
    self.maxT = self.physics_specs.nondimensionalize(maxT)
    self.dTy = self.physics_specs.nondimensionalize(dTy)
    self.dThz = self.physics_specs.nondimensionalize(dThz)
    _, sin_lat = self.coords.horizontal.nodal_mesh
    self.lat = np.arcsin(sin_lat)
    levels = coords.vertical
    if not isinstance(levels, hybrid_coordinates.HybridCoordinates):
      raise ValueError('Levels must be a HybridCoordinates.')
    nondim_a_boundaries = physics_specs.nondimensionalize(
        levels.a_boundaries * hpa_quantity
    )
    nondim_levels = hybrid_coordinates.HybridCoordinates(
        nondim_a_boundaries, levels.b_boundaries  # pyrefly: ignore[bad-argument-type]
    )
    self.nondim_coords = dataclasses.replace(coords, vertical=nondim_levels)

  def equilibrium_temperature(self, pressure: jnp.ndarray) -> jnp.ndarray:
    """Computes the equilibrium temperature profile."""
    p_over_p0 = pressure / self.p0
    temperature = p_over_p0**self.physics_specs.kappa * (
        self.maxT
        - self.dTy * np.sin(self.lat) ** 2
        - self.dThz * jnp.log(p_over_p0) * np.cos(self.lat) ** 2
    )
    return jnp.maximum(self.minT, temperature)

  def explicit_terms(
      self, state: primitive_equations.State
  ) -> primitive_equations.State:
    """Computes explicit tendencies due to Held-Suarez forcing."""
    aux_state = primitive_equations.compute_diagnostic_state_hybrid(
        state=state, coords=self.nondim_coords
    )

    nodal_log_surface_pressure = self.coords.horizontal.to_nodal(
        state.log_surface_pressure
    )
    nodal_surface_pressure = jnp.exp(nodal_log_surface_pressure)

    # Pressure at layer centers, with shape (levels, latitude, longitude)
    pressure = self.nondim_coords.vertical.pressure_centers(
        nodal_surface_pressure
    )

    # Here we use effective sigma, to match implementation of Held-Suarez for
    # Sigma cordinates. This effectively results in varying sigma values.
    sigma = pressure / nodal_surface_pressure

    # Rayleigh damping term, kv, is now a 3D field.
    kv_coeff = self.kf * jnp.maximum(
        0, (sigma - self.sigma_b) / (1 - self.sigma_b)
    )

    # Nodal velocity tendencies
    # here nodal velocity includes 1/cos(lat) factor that will be removed at
    # the end when we compute divergence and vorticity tendncies using
    # curl_cos_lat and div_cos_lat.
    nodal_velocity_tendency = jax.tree.map(
        lambda x: -kv_coeff * x / self.coords.horizontal.cos_lat**2,
        aux_state.cos_lat_u,
    )

    # Newtonian cooling term, kt, is also a 3D field.
    cutoff = jnp.maximum(0, (sigma - self.sigma_b) / (1 - self.sigma_b))
    kt_coeff = self.ka + (self.ks - self.ka) * (cutoff * np.cos(self.lat) ** 4)

    # Nodal temperature tendency
    nodal_temperature = (
        self.reference_temperature[:, np.newaxis, np.newaxis]
        + aux_state.temperature_variation
    )
    Teq = self.equilibrium_temperature(pressure)
    nodal_temperature_tendency = -kt_coeff * (nodal_temperature - Teq)

    # Convert to modal
    temperature_tendency = self.coords.horizontal.to_modal(
        nodal_temperature_tendency
    )
    velocity_tendency = self.coords.horizontal.to_modal(nodal_velocity_tendency)
    vorticity_tendency = self.coords.horizontal.curl_cos_lat(velocity_tendency)
    divergence_tendency = self.coords.horizontal.div_cos_lat(velocity_tendency)

    # Zero log_surface_pressure tendency
    log_surface_pressure_tendency = jnp.zeros_like(state.log_surface_pressure)

    return primitive_equations.State(
        vorticity=vorticity_tendency,  # pyrefly: ignore[unexpected-keyword]
        divergence=divergence_tendency,  # pyrefly: ignore[unexpected-keyword]
        temperature_variation=temperature_tendency,  # pyrefly: ignore[unexpected-keyword]
        log_surface_pressure=log_surface_pressure_tendency,  # pyrefly: ignore[unexpected-keyword]
    )


# Deprecated alias for backwards compatibility.
HeldSuarezForcing = HeldSuarezForcingSigma
