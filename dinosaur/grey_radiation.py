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

"""Grey (semi-grey, two-stream) radiative transfer for giant-planet GCMs.

This ports the "Schneider & Liu" giant-planet configuration of Isca's
`two_stream_gray_rad` scheme (Schneider, T., and J. Liu, 2009: "Formation of
jets and equatorial superrotation on Jupiter." J. Atmos. Sci., 66, 579-601),
together with its constant internal-heat-flux lower boundary condition, which
substitutes for a real surface on a planet that doesn't have one.

Longwave and shortwave optical depth both follow a power law in pressure,
non-dimensionalized by a reference pressure `p0`:

  tau_sw(p) = sw_tau_0 * (p / p0) ** sw_tau_exponent
  tau_lw(p) = lw_tau_0 * (p / p0) ** lw_tau_exponent

Shortwave flux is a pointwise function of pressure (no scattering between
layers other than through the two constant delta-Eddington correction
factors `scattering_albedo` and `ga_asym`, derived from
`single_scattering_albedo` and `back_scatter`). Longwave flux requires
solving the two-stream recurrence

  flux(k+1) = flux(k) * dtrans(k) + b(k) * (1 - dtrans(k))

separately downward (top of atmosphere to bottom) and upward (bottom to top),
where `dtrans(k)` is the transmissivity of layer `k` and `b(k)` is its
blackbody emission. The "surface" at the bottom of the domain has zero heat
capacity: it instantly re-emits all longwave and shortwave flux it absorbs,
plus a small constant internal heat flux (`flux_heat_gp`, representing a giant
planet's interior heat) is deposited directly as a temperature tendency in
the bottom-most layer.
"""

from dinosaur import coordinate_systems
from dinosaur import primitive_equations
from dinosaur import scales
from dinosaur import time_integration
from dinosaur import typing
from dinosaur import units as units_lib

import jax
import jax.numpy as jnp
import numpy as np

Quantity = typing.Quantity

# pylint: disable=invalid-name


def _affine_scan(a: jnp.ndarray, c: jnp.ndarray, x0: jnp.ndarray,
                  reverse: bool = False) -> jnp.ndarray:
  """Solves the affine recurrence `x[k+1] = a[k] * x[k] + c[k]` along axis 0.

  Args:
    a, c: arrays of shape `[n, ...]` giving the recurrence coefficients for
      `n` steps.
    x0: the boundary value the recurrence starts from, of shape `a.shape[1:]`.
    reverse: if False, the recurrence runs forward (`x[0] := x0`, producing
      `x[1], ..., x[n]`, i.e. `n` outputs aligned with `a[0], ..., a[n-1]`).
      If True, it runs backward (`x[n] := x0`, producing `x[n-1], ..., x[0]`,
      returned in the same `a`-aligned order as the forward case, i.e.
      `result[k]` corresponds to the value "above" `a[k]`/`c[k]`).

  Returns:
    An array of shape `a.shape` with the `n` computed values of the
    recurrence (not including the `x0` boundary itself).
  """
  def step(carry, ac):
    a_k, c_k = ac
    new_carry = a_k * carry + c_k
    return new_carry, new_carry

  _, ys = jax.lax.scan(step, x0, (a, c), reverse=reverse)
  return ys


def shortwave_tau(
    p_half: jnp.ndarray, sw_tau_0: float, sw_tau_exponent: float, p0: float
) -> jnp.ndarray:
  """Shortwave optical depth as a function of half-level pressure."""
  return sw_tau_0 * (p_half / p0) ** sw_tau_exponent


def shortwave_down_flux(
    p_half: jnp.ndarray,
    insolation: jnp.ndarray,
    sw_tau_0: float,
    sw_tau_exponent: float,
    p0: float,
    scattering_albedo: float,
    ga_asym: float,
) -> jnp.ndarray:
  """Downward shortwave flux at each half-level (no upward scan needed)."""
  tau_sw = shortwave_tau(p_half, sw_tau_0, sw_tau_exponent, p0)
  return insolation * (1 - scattering_albedo) * jnp.exp(-ga_asym * tau_sw)


def longwave_tau(
    p_half: jnp.ndarray, lw_tau_0: float, lw_tau_exponent: float, p0: float
) -> jnp.ndarray:
  """Longwave optical depth as a function of half-level pressure."""
  return lw_tau_0 * (p_half / p0) ** lw_tau_exponent


def longwave_dtrans(tau_lw: jnp.ndarray) -> jnp.ndarray:
  """Transmissivity of each layer, from its bounding half-level optical depths."""
  return jnp.exp(-(tau_lw[1:] - tau_lw[:-1]))


def longwave_fluxes(
    blackbody: jnp.ndarray,
    dtrans: jnp.ndarray,
    sw_down_surf: jnp.ndarray,
    surface_albedo: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Solves the up/down two-stream longwave flux recurrences.

  Args:
    blackbody: `stefan_boltzmann * T**4` at layer centers, shape `[n, ...]`.
    dtrans: layer transmissivities from `longwave_dtrans`, shape `[n, ...]`.
    sw_down_surf: downward shortwave flux at the bottom boundary.
    surface_albedo: albedo of the zero-heat-capacity bottom boundary (0 for a
      giant planet with no real surface).

  Returns:
    `(lw_down, lw_up)`, each of shape `[n + 1, ...]` (one value per
    half-level, top of atmosphere to bottom boundary).
  """
  emission = blackbody * (1 - dtrans)

  down_rest = _affine_scan(dtrans, emission, jnp.zeros_like(blackbody[0]))
  lw_down = jnp.concatenate([jnp.zeros_like(down_rest[:1]), down_rest], axis=0)

  # The bottom boundary has zero heat capacity: it instantly re-emits all
  # longwave and shortwave flux it absorbs.
  b_surf = lw_down[-1] + sw_down_surf * (1 - surface_albedo)

  up_rest = _affine_scan(dtrans, emission, b_surf, reverse=True)
  lw_up = jnp.concatenate([up_rest, b_surf[jnp.newaxis]], axis=0)

  return lw_down, lw_up


def radiative_heating_rate(
    rad_flux: jnp.ndarray, p_half: jnp.ndarray, gravity: float, cp: float
) -> jnp.ndarray:
  """Heating rate at layer centers from the net upward flux at half-levels."""
  return gravity * (rad_flux[1:] - rad_flux[:-1]) / (
      cp * (p_half[1:] - p_half[:-1])
  )


def internal_heat_tendency(
    p_half: jnp.ndarray, flux_heat_gp: float, gravity: float, cp: float
) -> jnp.ndarray:
  """Temperature tendency from a constant internal heat flux.

  The flux is deposited entirely in the bottom-most layer, matching Isca's
  `gp_surface_flux` (a fixed interior heat flux, not a true radiative
  boundary condition).
  """
  n_layers = p_half.shape[0] - 1
  bottom_dp = p_half[-1] - p_half[-2]
  bottom_tendency = gravity * flux_heat_gp / (cp * bottom_dp)
  zeros = jnp.zeros((n_layers,) + bottom_tendency.shape)
  return zeros.at[-1].set(bottom_tendency)


class GiantPlanetGreyRadiation(time_integration.ExplicitODE):
  """Two-stream grey-gas radiative heating for a giant planet.

  See module docstring for the physics. Composes with `PrimitiveEquations`
  the same way `held_suarez.LianShowmanForcing` does, via
  `time_integration.compose_equations`.
  """

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      physics_specs: units_lib.SimUnitsProtocol,
      reference_temperature: typing.Array,
      p0: Quantity = 1e5 * scales.units.pascal,  # pyrefly: ignore[unsupported-operation]
      sw_tau_0: float = 3.0,
      sw_tau_exponent: float = 1.0,
      lw_tau_0: float = 80.0,
      lw_tau_exponent: float = 2.0,
      single_scattering_albedo: float = 0.8,
      back_scatter: float = 0.398,
      surface_albedo: float = 0.0,
      solar_constant: Quantity = 50.7 * scales.units.W / scales.units.m**2,  # pyrefly: ignore[unsupported-operation]
      flux_heat_gp: Quantity = 5.7 * scales.units.W / scales.units.m**2,  # pyrefly: ignore[unsupported-operation]
  ):
    """Initialize GiantPlanetGreyRadiation.

    Args:
      coords: horizontal and vertical descritization.
      physics_specs: object holding physical constants and definition of
        custom units to use for initialization of the state.
      reference_temperature: horizontal reference temperature at all
        altitudes.
      p0: reference pressure used to non-dimensionalize the optical-depth
        power laws.
      sw_tau_0: shortwave optical depth at `p0`.
      sw_tau_exponent: power-law exponent of shortwave optical depth in
        pressure.
      lw_tau_0: longwave optical depth at `p0`.
      lw_tau_exponent: power-law exponent of longwave optical depth in
        pressure.
      single_scattering_albedo: single-scattering albedo used (with
        `back_scatter`) to derive the delta-Eddington shortwave scattering
        correction.
      back_scatter: backscatter fraction used to derive the delta-Eddington
        asymmetry factor.
      surface_albedo: albedo of the zero-heat-capacity bottom boundary. 0 for
        a giant planet with no real surface.
      solar_constant: mean incident stellar flux at the top of the
        atmosphere.
      flux_heat_gp: constant internal heat flux deposited in the bottom-most
        layer, representing a giant planet's interior heat.
    """
    self.coords = coords
    self.physics_specs = physics_specs
    self.reference_temperature = reference_temperature
    self.p0 = physics_specs.nondimensionalize(p0)
    self.sw_tau_0 = sw_tau_0
    self.sw_tau_exponent = sw_tau_exponent
    self.lw_tau_0 = lw_tau_0
    self.lw_tau_exponent = lw_tau_exponent
    self.surface_albedo = surface_albedo
    self.stefan_boltzmann = physics_specs.nondimensionalize(
        scales.STEFAN_BOLTZMANN
    )
    self.flux_heat_gp = physics_specs.nondimensionalize(flux_heat_gp)

    # Delta-Eddington corrections for shortwave scattering (Isca's `gp_albedo`
    # and `Ga_asym`; renamed here to avoid colliding with `surface_albedo`,
    # a physically distinct quantity that happens to share Isca's name).
    g_asym = 1 - 2 * back_scatter
    sqrt_term_scattering = np.sqrt(1 - g_asym * single_scattering_albedo)
    sqrt_term_absorption = np.sqrt(1 - single_scattering_albedo)
    self.scattering_albedo = (
        (sqrt_term_scattering - sqrt_term_absorption)
        / (sqrt_term_scattering + sqrt_term_absorption)
    )
    self.ga_asym = 2 * sqrt_term_absorption * sqrt_term_scattering

    _, sin_lat = self.coords.horizontal.nodal_mesh
    self.lat = np.arcsin(sin_lat)
    solar_constant_nondim = physics_specs.nondimensionalize(solar_constant)
    # Non-seasonal, zonally-uniform insolation (no diurnal/seasonal cycle).
    self.insolation = (solar_constant_nondim / np.pi) * np.cos(self.lat)

    self.boundaries = self.coords.vertical.boundaries[:, np.newaxis, np.newaxis]

  def half_level_pressure(self, nodal_surface_pressure: jnp.ndarray) -> jnp.ndarray:
    return self.boundaries * nodal_surface_pressure

  def explicit_terms(
      self, state: primitive_equations.State
  ) -> primitive_equations.State:
    """Computes explicit temperature tendencies due to grey-gas radiation."""
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
    p_half = self.half_level_pressure(nodal_surface_pressure)

    sw_down = shortwave_down_flux(
        p_half,
        self.insolation,
        self.sw_tau_0,
        self.sw_tau_exponent,
        self.p0,
        self.scattering_albedo,
        self.ga_asym,
    )
    tau_lw = longwave_tau(p_half, self.lw_tau_0, self.lw_tau_exponent, self.p0)
    dtrans = longwave_dtrans(tau_lw)
    blackbody = self.stefan_boltzmann * temperature**4
    lw_down, lw_up = longwave_fluxes(
        blackbody, dtrans, sw_down[-1], self.surface_albedo
    )
    sw_up = self.surface_albedo * sw_down[-1]

    rad_flux = (lw_up - lw_down) + (sw_up - sw_down)
    nodal_temperature_tendency = radiative_heating_rate(
        rad_flux, p_half, self.physics_specs.g, self.physics_specs.Cp
    )
    nodal_temperature_tendency += internal_heat_tendency(
        p_half, self.flux_heat_gp, self.physics_specs.g, self.physics_specs.Cp
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
