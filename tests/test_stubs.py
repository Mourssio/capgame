"""Tests for deliberate placeholder modules (sfe, mpe)."""

from __future__ import annotations

import pytest

from capgame.game.cournot import Firm, LinearDemand


class TestSFEStub:
    def test_sfe_raises_informative_error(self) -> None:
        from capgame.game.sfe import solve_sfe

        demand = LinearDemand(a=100.0, b=1.0)
        firms = [Firm(marginal_cost=10.0, capacity=20.0)]
        with pytest.raises(NotImplementedError, match="Phase 2b"):
            solve_sfe(demand, firms)


class TestMPEForwarder:
    def test_mpe_reexports_backward_induction(self) -> None:
        from capgame.game.mpe import backward_induction
        from capgame.optimization.sdp import backward_induction as bi

        assert backward_induction is bi


class TestMechanismProtocol:
    def test_concrete_mechanisms_satisfy_protocol(self) -> None:
        from capgame.mechanisms.base import Mechanism
        from capgame.mechanisms.capacity_payment import CapacityPayment
        from capgame.mechanisms.energy_only import EnergyOnly
        from capgame.mechanisms.forward_capacity import (
            ForwardCapacityMarket,
            ProcurementCurve,
        )
        from capgame.mechanisms.reliability_options import ReliabilityOption

        mechs: list[Mechanism] = [
            EnergyOnly(),
            CapacityPayment(rho=5.0),
            ForwardCapacityMarket(curve=ProcurementCurve(cap_target=50.0, slope=1.0)),
            ReliabilityOption(premium=1.0, strike_price=50.0),
        ]
        for m in mechs:
            assert isinstance(m, Mechanism)
