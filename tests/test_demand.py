"""Tests for the Markov demand chain."""

from __future__ import annotations

import numpy as np
import pytest

from capgame.stochastic.demand import DemandState, MarkovChain, three_state_chain


class TestDemandState:
    def test_validates(self) -> None:
        with pytest.raises(ValueError):
            DemandState(name="bad", intercept=-1.0)
        with pytest.raises(ValueError):
            DemandState(name="bad", intercept=10.0, slope=0.0)

    def test_effective_peak_defaults_to_choke(self) -> None:
        s = DemandState(name="x", intercept=100.0, slope=2.0)
        assert s.effective_peak_load == pytest.approx(50.0)


class TestMarkovChain:
    def test_rejects_non_stochastic_matrix(self) -> None:
        with pytest.raises(ValueError):
            MarkovChain(
                states=[DemandState("a", 1.0), DemandState("b", 1.0)],
                transition_matrix=np.array([[0.5, 0.6], [0.5, 0.5]]),
            )

    def test_rejects_wrong_shape(self) -> None:
        with pytest.raises(ValueError):
            MarkovChain(
                states=[DemandState("a", 1.0), DemandState("b", 1.0)],
                transition_matrix=np.eye(3),
            )

    def test_identity_is_stationary(self) -> None:
        states = [DemandState("a", 1.0), DemandState("b", 1.0)]
        chain = MarkovChain(states=states, transition_matrix=np.eye(2))
        pi = chain.stationary_distribution()
        assert np.isclose(pi.sum(), 1.0)

    def test_symmetric_chain_has_uniform_stationary(self) -> None:
        chain = three_state_chain()
        pi = chain.stationary_distribution()
        np.testing.assert_allclose(pi, np.ones(3) / 3, atol=1e-8)

    def test_distribution_at_matches_powers(self) -> None:
        chain = three_state_chain(persistence=0.7)
        P = chain.transition_matrix
        pi0 = chain.initial_distribution
        expected = pi0 @ np.linalg.matrix_power(P, 5)
        np.testing.assert_allclose(chain.distribution_at(5), expected, atol=1e-12)

    def test_sample_respects_transitions(self) -> None:
        chain = three_state_chain(persistence=0.9)
        rng = np.random.default_rng(42)
        path = chain.sample(n_steps=10_000, rng=rng, start_state=1)
        assert path.shape == (10_001,)
        assert path[0] == 1
        assert path.min() >= 0
        assert path.max() <= 2

    def test_stationary_matches_long_run_frequency(self) -> None:
        chain = three_state_chain(persistence=0.6)
        rng = np.random.default_rng(0)
        path = chain.sample(n_steps=50_000, rng=rng)
        freq = np.bincount(path, minlength=3) / path.size
        np.testing.assert_allclose(freq, chain.stationary_distribution(), atol=0.02)
