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

"""Tests for dry_convection."""

from absl.testing import absltest
from absl.testing import parameterized

from dinosaur import coordinate_systems
from dinosaur import dry_convection
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


class ParcelTemperatureTest(parameterized.TestCase):

  def test_isothermal_gamma_zero_reproduces_isothermal_column(self):
    n, shape = 10, (2, 3)
    p_full = np.geomspace(100.0, 1e5, n)[:, np.newaxis, np.newaxis] * np.ones(
        (1,) + shape
    )
    temperature = 200.0 * np.ones((n,) + shape)
    tp = dry_convection.parcel_temperature(
        jnp.asarray(temperature), jnp.asarray(p_full), gamma=0.0, kappa=0.29
    )
    np.testing.assert_allclose(tp, 200.0, rtol=1e-6)

  def test_dry_adiabat_gamma_one_conserves_potential_temperature(self):
    n, shape = 10, (2, 3)
    p_full = np.geomspace(100.0, 1e5, n)
    kappa = 0.29
    # A column already exactly on the dry adiabat relative to the bottom.
    temperature = 200.0 * (p_full / p_full[-1]) ** kappa
    temperature = temperature[:, np.newaxis, np.newaxis] * np.ones((1,) + shape)
    p_full_b = p_full[:, np.newaxis, np.newaxis] * np.ones((1,) + shape)

    tp = dry_convection.parcel_temperature(
        jnp.asarray(temperature), jnp.asarray(p_full_b), gamma=1.0, kappa=kappa
    )
    np.testing.assert_allclose(tp, temperature, rtol=1e-5)


class ConvectingRegionTest(parameterized.TestCase):

  def test_finds_contiguous_bottom_attached_region(self):
    # Unstable (tp >= T) for the bottom 3 layers, stable above.
    temperature = jnp.array([300.0, 300.0, 100.0, 100.0, 100.0])[
        :, jnp.newaxis, jnp.newaxis
    ]
    tp = jnp.array([50.0, 50.0, 150.0, 150.0, 150.0])[:, jnp.newaxis, jnp.newaxis]
    region = dry_convection.convecting_region(tp, temperature)
    np.testing.assert_array_equal(
        region[:, 0, 0], [False, False, True, True, True]
    )

  def test_ignores_disconnected_unstable_layer_above_a_stable_gap(self):
    # Unstable at the very top (index 0) and at the bottom (index 3), but
    # separated by a stable layer (index 1-2); only the bottom-attached
    # region should be flagged.
    temperature = jnp.array([50.0, 300.0, 300.0, 100.0])[:, jnp.newaxis, jnp.newaxis]
    tp = jnp.array([150.0, 50.0, 50.0, 150.0])[:, jnp.newaxis, jnp.newaxis]
    region = dry_convection.convecting_region(tp, temperature)
    np.testing.assert_array_equal(region[:, 0, 0], [False, False, False, True])


class AdjustmentTendencyTest(parameterized.TestCase):

  def test_conserves_pressure_weighted_column_energy(self):
    rng = np.random.RandomState(0)
    n, shape = 12, (2, 3)
    p_half = np.linspace(0, 1, n + 1)[:, np.newaxis, np.newaxis] * 1e5 * np.ones(
        (1,) + shape
    )
    dp_half = p_half[1:] - p_half[:-1]
    temperature = rng.uniform(100, 300, size=(n,) + shape)
    p_full = 0.5 * (p_half[1:] + p_half[:-1])

    tp = dry_convection.parcel_temperature(
        jnp.asarray(temperature), jnp.asarray(p_full), gamma=1.0, kappa=0.29
    )
    tau = 21600.0
    tendency = dry_convection.adjustment_tendency(
        jnp.asarray(temperature), tp, jnp.asarray(dp_half), tau
    )
    weighted_sum = jnp.sum(dp_half * tendency, axis=0)
    np.testing.assert_allclose(weighted_sum, 0.0, atol=1e-3)

  def test_zero_tendency_when_already_on_adiabat(self):
    n, shape = 10, (1, 1)
    p_full = np.geomspace(100.0, 1e5, n)
    kappa = 0.29
    temperature = 200.0 * (p_full / p_full[-1]) ** kappa
    temperature = temperature[:, np.newaxis, np.newaxis] * np.ones((1,) + shape)
    p_full_b = p_full[:, np.newaxis, np.newaxis] * np.ones((1,) + shape)
    dp_half = np.ones((n,) + shape)

    tp = dry_convection.parcel_temperature(
        jnp.asarray(temperature), jnp.asarray(p_full_b), gamma=1.0, kappa=kappa
    )
    tendency = dry_convection.adjustment_tendency(
        jnp.asarray(temperature), tp, jnp.asarray(dp_half), tau=21600.0
    )
    np.testing.assert_allclose(tendency, 0.0, atol=1e-4)


class DryConvectiveAdjustmentTest(parameterized.TestCase):

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

    convection = dry_convection.DryConvectiveAdjustment(
        coords=coords,
        physics_specs=physics_specs,
        reference_temperature=ref_temps,
    )

    explicit_terms = convection.explicit_terms(state)
    np.testing.assert_allclose(explicit_terms.vorticity, 0)
    np.testing.assert_allclose(explicit_terms.divergence, 0)
    np.testing.assert_allclose(explicit_terms.log_surface_pressure, 0)
    self.assertEqual(
        explicit_terms.temperature_variation.shape, coords.modal_shape
    )
    # An isothermal column is *stable* relative to a dry adiabat (gamma=1;
    # its lapse rate is shallower than adiabatic), so this should trigger no
    # adjustment at all -- covered more directly by AdjustmentTendencyTest
    # and ConvectingRegionTest above; here we're just checking the class
    # wires shapes/dtypes together correctly end-to-end.
    np.testing.assert_allclose(explicit_terms.temperature_variation, 0)


if __name__ == '__main__':
  absltest.main()
