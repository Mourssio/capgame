"""Tests for the renewable-availability chain.

These verify that :class:`RenewableState` carries the right semantics
(capacity factor in ``[0, 1]``, linear conversion to MW output) and that
the toy chain returned by :func:`simple_two_state_renewables` is a
well-formed :class:`MarkovChain[RenewableState]`.
"""

from __future__ import annotations

import numpy as np
import pytest

from capgame.stochastic.demand import MarkovChain
from capgame.stochastic.renewables import RenewableState, simple_two_state_renewables


class TestRenewableState:
    def test_rejects_wind_cf_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            RenewableState(name="bad", wind_cf=1.1, solar_cf=0.2)
        with pytest.raises(ValueError):
            RenewableState(name="bad", wind_cf=-0.1, solar_cf=0.2)

    def test_rejects_solar_cf_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            RenewableState(name="bad", wind_cf=0.2, solar_cf=1.5)

    def test_available_output_linear(self) -> None:
        state = RenewableState(name="s", wind_cf=0.4, solar_cf=0.2)
        assert state.available_output(100.0, 50.0) == pytest.approx(0.4 * 100 + 0.2 * 50)

    def test_available_output_rejects_negative_capacity(self) -> None:
        state = RenewableState(name="s", wind_cf=0.4, solar_cf=0.2)
        with pytest.raises(ValueError):
            state.available_output(-10.0, 50.0)

    def test_frozen(self) -> None:
        import dataclasses

        state = RenewableState(name="s", wind_cf=0.4, solar_cf=0.2)
        with pytest.raises(dataclasses.FrozenInstanceError):
            state.wind_cf = 0.5  # type: ignore[misc]


class TestSimpleTwoStateRenewables:
    def test_returns_markov_chain_of_renewable_state(self) -> None:
        chain = simple_two_state_renewables()
        assert isinstance(chain, MarkovChain)
        assert chain.n_states == 4
        for s in chain.states:
            assert isinstance(s, RenewableState)

    def test_state_payloads_match_expected_grid(self) -> None:
        chain = simple_two_state_renewables(
            low_wind=0.1, high_wind=0.5, low_solar=0.2, high_solar=0.4
        )
        names = {s.name for s in chain.states}
        assert names == {"wL_sL", "wL_sH", "wH_sL", "wH_sH"}
        by_name = {s.name: s for s in chain.states}
        assert by_name["wL_sL"].wind_cf == 0.1
        assert by_name["wH_sH"].solar_cf == 0.4

    def test_transition_matrix_row_stochastic(self) -> None:
        chain = simple_two_state_renewables(correlation=0.7)
        P = chain.transition_matrix
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-12)
        assert np.all(P >= 0.0)

    def test_positive_correlation_biases_diagonal(self) -> None:
        neutral = simple_two_state_renewables(correlation=0.0).transition_matrix
        biased = simple_two_state_renewables(correlation=0.5).transition_matrix
        assert np.all(np.diag(biased) >= np.diag(neutral) - 1e-12)
        assert np.trace(biased) > np.trace(neutral)

    def test_rejects_correlation_outside_open_unit_interval(self) -> None:
        with pytest.raises(ValueError):
            simple_two_state_renewables(correlation=1.0)
        with pytest.raises(ValueError):
            simple_two_state_renewables(correlation=-1.0)

    def test_does_not_overload_demand_state(self) -> None:
        """Regression: the chain's payload must be RenewableState, not DemandState.

        The previous implementation stuffed capacity factors into the
        DemandState.intercept field with a ``+ 1.0`` fudge to satisfy
        ``intercept > 0`` validation. This test fixes the semantic
        separation in place.
        """
        from capgame.stochastic.demand import DemandState

        chain = simple_two_state_renewables()
        for s in chain.states:
            assert not isinstance(s, DemandState)


class TestGenericMarkovChain:
    """The chain machinery should work identically regardless of payload type."""

    def test_sampling_works_with_renewable_payload(self) -> None:
        chain = simple_two_state_renewables(correlation=0.3)
        rng = np.random.default_rng(0)
        path = chain.sample(n_steps=500, rng=rng)
        assert path.shape == (501,)
        assert set(path.tolist()) <= {0, 1, 2, 3}

    def test_stationary_distribution_sums_to_one(self) -> None:
        chain = simple_two_state_renewables(correlation=0.3)
        pi = chain.stationary_distribution()
        assert pi.sum() == pytest.approx(1.0, abs=1e-10)
        assert np.all(pi >= -1e-12)

    def test_state_accessor_returns_typed_payload(self) -> None:
        chain = simple_two_state_renewables()
        s0 = chain.state(0)
        assert isinstance(s0, RenewableState)
        assert s0.name == "wL_sL"
