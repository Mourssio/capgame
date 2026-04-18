"""Tests for adequacy metrics and forced-outage convolution."""

from __future__ import annotations

import numpy as np
import pytest

from capgame.adequacy._validation import (
    validate_demand_pmf,
    validate_fleet,
    validate_scaling,
)
from capgame.adequacy.eue import (
    expected_unserved_energy,
    expected_unserved_energy_monte_carlo,
)
from capgame.adequacy.lole import (
    lole_from_capacity_distribution,
    loss_of_load_expectation,
    loss_of_load_probability,
)
from capgame.adequacy.reserve_margin import (
    capacity_required,
    reserve_margin,
    system_capacity,
)
from capgame.stochastic.outages import (
    ForcedOutage,
    effective_capacity_distribution,
    sample_effective_capacity,
)


class TestReserveMargin:
    def test_simple(self) -> None:
        assert reserve_margin(120.0, 100.0) == pytest.approx(0.20)

    def test_negative_margin(self) -> None:
        assert reserve_margin(80.0, 100.0) == pytest.approx(-0.20)

    def test_zero_peak_raises(self) -> None:
        with pytest.raises(ValueError):
            reserve_margin(100.0, 0.0)

    def test_system_capacity(self) -> None:
        assert system_capacity([10.0, 20.0, 30.0]) == 60.0


class TestCapacityRequired:
    def test_fifteen_percent_margin(self) -> None:
        assert capacity_required(1000.0, 0.15) == pytest.approx(1150.0)

    def test_zero_margin_returns_peak(self) -> None:
        assert capacity_required(500.0, 0.0) == pytest.approx(500.0)

    def test_is_inverse_of_reserve_margin(self) -> None:
        peak, margin = 800.0, 0.25
        cap = capacity_required(peak, margin)
        assert reserve_margin(cap, peak) == pytest.approx(margin)

    def test_rejects_negative_margin(self) -> None:
        with pytest.raises(ValueError):
            capacity_required(1000.0, -0.01)

    def test_rejects_zero_peak(self) -> None:
        with pytest.raises(ValueError):
            capacity_required(0.0, 0.15)


class TestValidators:
    def test_demand_pmf_rejects_empty(self) -> None:
        with pytest.raises(ValueError):
            validate_demand_pmf([])

    def test_demand_pmf_rejects_negative(self) -> None:
        with pytest.raises(ValueError):
            validate_demand_pmf([(100.0, -0.1), (200.0, 1.1)])

    def test_demand_pmf_rejects_non_unit_sum(self) -> None:
        with pytest.raises(ValueError):
            validate_demand_pmf([(100.0, 0.4), (200.0, 0.4)])

    def test_fleet_rejects_mismatched_length(self) -> None:
        with pytest.raises(ValueError):
            validate_fleet([100.0, 50.0], [0.1])

    def test_fleet_rejects_negative_capacity(self) -> None:
        with pytest.raises(ValueError):
            validate_fleet([100.0, -1.0], [0.1, 0.1])

    def test_fleet_rejects_bad_outage_rate(self) -> None:
        with pytest.raises(ValueError):
            validate_fleet([100.0], [1.0])

    def test_scaling_rejects_non_positive(self) -> None:
        with pytest.raises(ValueError):
            validate_scaling(0.0)


class TestForcedOutage:
    def test_availability(self) -> None:
        assert ForcedOutage(outage_rate=0.05).availability == pytest.approx(0.95)

    def test_rejects_invalid(self) -> None:
        with pytest.raises(ValueError):
            ForcedOutage(outage_rate=1.0)


class TestCapacityConvolution:
    def test_single_unit(self) -> None:
        support, probs = effective_capacity_distribution([100.0], [0.1])
        assert set(support.tolist()) == {0.0, 100.0}
        idx_hi = np.argmax(support)
        assert probs[idx_hi] == pytest.approx(0.9)

    def test_probabilities_sum_to_one(self) -> None:
        caps = [100.0, 50.0, 25.0]
        fors = [0.1, 0.05, 0.2]
        _, probs = effective_capacity_distribution(caps, fors)
        assert probs.sum() == pytest.approx(1.0)

    def test_identical_units_have_binomial_distribution(self) -> None:
        """Two (and five) identical units: COPT is Binomial(N, 1-f)."""
        from math import comb

        for N in (2, 5):
            cap = 10.0
            f = 0.2
            support, probs = effective_capacity_distribution([cap] * N, [f] * N)
            counts = (support / cap).round().astype(int)
            for c, p in zip(counts, probs, strict=True):
                expected = comb(N, int(c)) * (1 - f) ** c * f ** (N - c)
                assert p == pytest.approx(expected, abs=1e-10)

    def test_rejects_large_fleet(self) -> None:
        with pytest.raises(ValueError):
            effective_capacity_distribution([1.0] * 25, [0.1] * 25)

    def test_monte_carlo_matches_exact(self) -> None:
        caps = [100.0, 50.0, 25.0]
        fors = [0.1, 0.05, 0.2]
        support, probs = effective_capacity_distribution(caps, fors)
        exact_mean = float((support * probs).sum())
        rng = np.random.default_rng(123)
        draws = sample_effective_capacity(caps, fors, 50_000, rng=rng)
        assert draws.mean() == pytest.approx(exact_mean, rel=0.01)


class TestLoLP:
    def test_zero_when_all_capacity_covers_peak(self) -> None:
        lolp = loss_of_load_probability([1000.0], [0.0], peak_load=500.0)
        assert lolp == 0.0

    def test_one_when_capacity_below_peak_with_certainty(self) -> None:
        lolp = loss_of_load_probability([100.0], [0.0], peak_load=500.0)
        assert lolp == pytest.approx(1.0)

    def test_matches_single_unit_outage_rate(self) -> None:
        lolp = loss_of_load_probability([100.0], [0.1], peak_load=50.0)
        assert lolp == pytest.approx(0.1)

    def test_hand_calculation_two_units(self) -> None:
        """Two 50 MW units, f=0.1 each, peak=75.

        Shortfall iff at least one unit is out: P = 1 - 0.9^2 = 0.19.
        """
        lolp = loss_of_load_probability([50.0, 50.0], [0.1, 0.1], peak_load=75.0)
        assert lolp == pytest.approx(1 - 0.81)

    def test_rejects_length_mismatch(self) -> None:
        with pytest.raises(ValueError):
            loss_of_load_probability([100.0, 50.0], [0.1], peak_load=50.0)


class TestLoLE:
    def test_weighted_over_demand(self) -> None:
        """Single unit, two demand states, hand calculation."""
        caps = [100.0]
        fors = [0.05]
        demand = [(50.0, 0.7), (150.0, 0.3)]
        lole = loss_of_load_expectation(caps, fors, demand)
        expected = 0.7 * 0.05 + 0.3 * 1.0
        assert lole == pytest.approx(expected)

    def test_periods_per_year_scales(self) -> None:
        caps = [100.0]
        fors = [0.05]
        demand = [(50.0, 1.0)]
        lole_hours = loss_of_load_expectation(caps, fors, demand, periods_per_year=8760.0)
        assert lole_hours == pytest.approx(0.05 * 8760.0)

    def test_from_distribution_matches_direct(self) -> None:
        caps = [100.0, 50.0]
        fors = [0.1, 0.05]
        demand = [(40.0, 0.5), (120.0, 0.5)]
        direct = loss_of_load_expectation(caps, fors, demand)
        support, probs = effective_capacity_distribution(caps, fors)
        indirect = lole_from_capacity_distribution(support, probs, demand)
        assert direct == pytest.approx(indirect)

    def test_zero_when_abundant_and_reliable(self) -> None:
        """Unlimited reliable capacity versus finite demand -> LOLE = 0."""
        lole = loss_of_load_expectation([10_000.0], [0.0], [(500.0, 0.5), (1000.0, 0.5)])
        assert lole == 0.0

    def test_rejects_bad_pmf(self) -> None:
        with pytest.raises(ValueError):
            loss_of_load_expectation([100.0], [0.0], [(50.0, 0.6), (80.0, 0.5)])

    def test_rejects_length_mismatch_at_boundary(self) -> None:
        with pytest.raises(ValueError):
            loss_of_load_expectation([100.0, 50.0], [0.1], [(50.0, 1.0)])

    def test_rejects_non_positive_scaling(self) -> None:
        with pytest.raises(ValueError):
            loss_of_load_expectation([100.0], [0.0], [(50.0, 1.0)], periods_per_year=0.0)


class TestEUE:
    def test_zero_when_capacity_always_covers_peak(self) -> None:
        eue = expected_unserved_energy([1000.0], [0.0], [(500.0, 1.0)])
        assert eue == 0.0

    def test_exact_shortfall(self) -> None:
        eue = expected_unserved_energy([100.0], [0.0], [(150.0, 1.0)])
        assert eue == pytest.approx(50.0)

    def test_hand_calculation_two_units(self) -> None:
        """Two 50 MW units, f=0.1 each, peak=75 with probability 1.

        Outcomes: both up (P=0.81, short 0), one up (P=2*0.09, short 25),
        both out (P=0.01, short 75).  EUE = 2 * 0.09 * 25 + 0.01 * 75 = 5.25.
        """
        eue = expected_unserved_energy([50.0, 50.0], [0.1, 0.1], [(75.0, 1.0)])
        assert eue == pytest.approx(4.5 + 0.75)

    def test_zero_when_abundant_and_reliable(self) -> None:
        eue = expected_unserved_energy([10_000.0], [0.0], [(500.0, 0.5), (1000.0, 0.5)])
        assert eue == 0.0

    def test_monte_carlo_matches_exact(self) -> None:
        caps = [100.0, 50.0, 25.0]
        fors = [0.1, 0.05, 0.2]
        demand = [(60.0, 0.4), (130.0, 0.6)]
        exact = expected_unserved_energy(caps, fors, demand)
        rng = np.random.default_rng(7)
        mc = expected_unserved_energy_monte_carlo(caps, fors, demand, n_samples=100_000, rng=rng)
        assert mc == pytest.approx(exact, rel=0.05, abs=0.2)

    def test_monte_carlo_converges_as_n_grows(self) -> None:
        """Standard error should shrink like 1/sqrt(n); error halves for 4x samples."""
        caps = [100.0, 50.0, 25.0]
        fors = [0.1, 0.05, 0.2]
        demand = [(60.0, 0.4), (130.0, 0.6)]
        exact = expected_unserved_energy(caps, fors, demand)

        def mc_err(n: int, seed: int) -> float:
            rng = np.random.default_rng(seed)
            est = expected_unserved_energy_monte_carlo(caps, fors, demand, n_samples=n, rng=rng)
            return abs(est - exact)

        trials = 10
        small_err = float(np.mean([mc_err(1_000, s) for s in range(trials)]))
        large_err = float(np.mean([mc_err(64_000, s + 100) for s in range(trials)]))
        assert large_err < small_err * 0.5

    def test_rejects_length_mismatch_at_boundary(self) -> None:
        with pytest.raises(ValueError):
            expected_unserved_energy([100.0, 50.0], [0.1], [(50.0, 1.0)])

    def test_rejects_zero_samples(self) -> None:
        with pytest.raises(ValueError):
            expected_unserved_energy_monte_carlo([100.0], [0.1], [(50.0, 1.0)], n_samples=0)
