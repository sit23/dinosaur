# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for grey_radiation."""

from absl.testing import absltest
from absl.testing import parameterized

from dinosaur import coordinate_systems
from dinosaur import grey_radiation
from dinosaur import primitive_equations
from dinosaur import primitive_equations_states
from dinosaur import scales
from dinosaur import sigma_coordinates
from dinosaur import spherical_harmonic
from dinosaur import xarray_utils

import jax
import jax.numpy as jnp
import numpy as np

jax.config.parse_flags_with_absl()


class AffineScanTest(parameterized.TestCase):

  def _reference_forward(self, a, c, x0):
    x = x0
    ys = []
    for a_k, c_k in zip(a, c):
      x = a_k * x + c_k
      ys.append(x)
    return np.stack(ys)

  def _reference_backward(self, a, c, x0):
    x = x0
    ys = [None] * len(a)
    for k in reversed(range(len(a))):
      x = a[k] * x + c[k]
      ys[k] = x
    return np.stack(ys)

  @parameterized.parameters(False, True)
  def test_matches_python_loop(self, reverse):
    rng = np.random.RandomState(0)
    n, shape = 6, (3, 4)
    a = rng.uniform(0.1, 0.9, size=(n,) + shape)
    c = rng.uniform(-1, 1, size=(n,) + shape)
    x0 = rng.uniform(-1, 1, size=shape)

    actual = grey_radiation._affine_scan(
        jnp.asarray(a), jnp.asarray(c), jnp.asarray(x0), reverse=reverse
    )
    expected = (
        self._reference_backward(a, c, x0)
        if reverse
        else self._reference_forward(a, c, x0)
    )
    # jax runs the scan in float32 by default; the reference python loop is
    # float64, so allow for float32-level rounding error over 6 steps.
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


class FluxEnergyConservationTest(parameterized.TestCase):

  def test_heating_rate_telescopes_to_flux_divergence(self):
    rng = np.random.RandomState(0)
    n, shape = 20, (2, 3)
    p0 = 25e5
    boundaries = np.linspace(0, 1, n + 1)
    surface_pressure = p0 * np.ones(shape)
    p_half = boundaries[:, np.newaxis, np.newaxis] * surface_pressure

    temperature = rng.uniform(150, 400, size=(n,) + shape)
    lat = np.linspace(-np.pi / 2, np.pi / 2, shape[-1])
    insolation = 50.0 / np.pi * np.cos(lat) * np.ones(shape)

    sw_tau_0, sw_tau_exponent = 3.0, 1.0
    lw_tau_0, lw_tau_exponent = 80.0, 2.0
    single_scattering_albedo, back_scatter = 0.8, 0.398
    surface_albedo = 0.0
    stefan_boltzmann = 5.670374e-8

    g_asym = 1 - 2 * back_scatter
    s1 = np.sqrt(1 - g_asym * single_scattering_albedo)
    s2 = np.sqrt(1 - single_scattering_albedo)
    scattering_albedo = (s1 - s2) / (s1 + s2)
    ga_asym = 2 * s2 * s1

    sw_down = grey_radiation.shortwave_down_flux(
        p_half, insolation, sw_tau_0, sw_tau_exponent, p0,
        scattering_albedo, ga_asym,
    )
    tau_lw = grey_radiation.longwave_tau(p_half, lw_tau_0, lw_tau_exponent, p0)
    dtrans = grey_radiation.longwave_dtrans(tau_lw)
    blackbody = stefan_boltzmann * temperature**4
    lw_down, lw_up = grey_radiation.longwave_fluxes(
        blackbody, dtrans, sw_down[-1], surface_albedo
    )
    sw_up = surface_albedo * sw_down[-1]
    rad_flux = (lw_up - lw_down) + (sw_up - sw_down)

    gravity, cp = 22.88, 13e3
    heating_rate = grey_radiation.radiative_heating_rate(
        rad_flux, p_half, gravity, cp
    )

    # Sum_k heating_rate[k] * cp * dp[k] / g must telescope exactly to the
    # net flux difference between the bottom and top boundaries.
    dp = p_half[1:] - p_half[:-1]
    total = np.sum(heating_rate * cp * dp / gravity, axis=0)
    expected = rad_flux[-1] - rad_flux[0]
    # float32 accumulation over `n` layers vs. a single subtraction.
    np.testing.assert_allclose(total, expected, rtol=1e-5)

  def test_internal_heat_goes_entirely_to_bottom_layer(self):
    n, shape = 10, (2, 2)
    p_half = np.linspace(0, 1, n + 1)[:, np.newaxis, np.newaxis] * 25e5 * np.ones(
        shape
    )
    gravity, cp, flux_heat_gp = 22.88, 13e3, 5.7
    tendency = grey_radiation.internal_heat_tendency(
        p_half, flux_heat_gp, gravity, cp
    )
    self.assertEqual(tendency.shape, (n,) + shape)
    np.testing.assert_allclose(tendency[:-1], 0)
    self.assertTrue(np.all(tendency[-1] > 0))


class GiantPlanetGreyRadiationTest(parameterized.TestCase):

  def test_explicit_terms_shapes_and_zero_tendencies(self):
    units = scales.units
    layers = 20
    coords = coordinate_systems.CoordinateSystem(
        horizontal=spherical_harmonic.Grid.T21(),
        vertical=sigma_coordinates.SigmaCoordinates.equidistant_log(layers, 11),
    )
    physics_specs = primitive_equations.PrimitiveEquationsSpecs.from_si()
    p0 = 25e5 * units.pascal

    initial_state_fn, aux_features = (
        primitive_equations_states.isothermal_rest_atmosphere(
            coords, physics_specs, p0=p0, p1=0.0 * units.pascal
        )
    )
    ref_temps = aux_features[xarray_utils.REF_TEMP_KEY]
    state = initial_state_fn(rng_key=jax.random.PRNGKey(0))

    radiation = grey_radiation.GiantPlanetGreyRadiation(
        coords=coords,
        physics_specs=physics_specs,
        reference_temperature=ref_temps,
        p0=p0,
    )

    explicit_terms = radiation.explicit_terms(state)
    np.testing.assert_allclose(explicit_terms.vorticity, 0)
    np.testing.assert_allclose(explicit_terms.divergence, 0)
    np.testing.assert_allclose(explicit_terms.log_surface_pressure, 0)
    self.assertEqual(
        explicit_terms.temperature_variation.shape, coords.modal_shape
    )
    # An isothermal start well above 0K should not produce NaNs/Infs anywhere.
    self.assertTrue(
        np.all(np.isfinite(explicit_terms.temperature_variation))
    )


if __name__ == '__main__':
  absltest.main()
