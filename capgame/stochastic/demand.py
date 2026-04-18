"""Finite-state Markov demand process.

Khalfallah's baseline uses a three-state chain {low, mid, high} with a 3x3
transition matrix. We generalize to an arbitrary finite state space. Each
state carries a :class:`DemandState` payload that the Cournot subgame
consumes — specifically the inverse-demand intercept.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class DemandState:
    """Per-state demand parameters.

    Parameters
    ----------
    name
        Human-readable label, e.g. ``"low"`` or ``"peak"``.
    intercept
        Inverse-demand intercept ``a`` in ``P(Q) = a - b*Q`` for this state.
    slope
        Inverse-demand slope ``b``. Kept state-independent in most applications
        but exposed here for flexibility (e.g. seasonal elasticity changes).
    peak_load
        Peak load in MW for reliability computations. Defaults to the choke
        quantity ``intercept/slope`` if not provided.
    """

    name: str
    intercept: float
    slope: float = 1.0
    peak_load: float | None = None

    def __post_init__(self) -> None:
        if self.intercept <= 0:
            raise ValueError(f"intercept must be positive, got {self.intercept}")
        if self.slope <= 0:
            raise ValueError(f"slope must be positive, got {self.slope}")

    @property
    def effective_peak_load(self) -> float:
        if self.peak_load is not None:
            return self.peak_load
        return self.intercept / self.slope


class MarkovChain:
    """Discrete-time, finite-state Markov chain.

    Parameters
    ----------
    states
        Ordered sequence of :class:`DemandState` instances.
    transition_matrix
        Row-stochastic 2-D array. ``transition_matrix[i, j]`` is the
        probability of moving from state ``i`` to state ``j`` in one step.
    initial_distribution
        Probability distribution over states at t=0. Defaults to the
        stationary distribution if not provided.
    """

    def __init__(
        self,
        states: Sequence[DemandState],
        transition_matrix: npt.ArrayLike,
        initial_distribution: npt.ArrayLike | None = None,
    ) -> None:
        if len(states) == 0:
            raise ValueError("At least one state is required.")
        P = np.asarray(transition_matrix, dtype=float)
        n = len(states)
        if P.ndim != 2 or P.shape[0] != n or P.shape[1] != n:
            raise ValueError(f"transition_matrix must be square of size {n}, got shape {P.shape}")
        if np.any(P < 0):
            raise ValueError("transition_matrix must be nonnegative.")
        row_sums = P.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-8):
            raise ValueError(f"transition_matrix rows must sum to 1, got row sums {row_sums}")

        self._states: tuple[DemandState, ...] = tuple(states)
        self._P: np.ndarray = P

        if initial_distribution is None:
            pi0 = self.stationary_distribution()
        else:
            pi0 = np.asarray(initial_distribution, dtype=float)
            if pi0.shape != (len(states),):
                raise ValueError(
                    f"initial_distribution must have shape ({len(states)},), got {pi0.shape}"
                )
            if not np.isclose(pi0.sum(), 1.0, atol=1e-8) or np.any(pi0 < 0):
                raise ValueError("initial_distribution must be a probability vector.")
        self._pi0 = pi0

    @property
    def n_states(self) -> int:
        return len(self._states)

    @property
    def states(self) -> tuple[DemandState, ...]:
        return self._states

    @property
    def transition_matrix(self) -> np.ndarray:
        return self._P.copy()

    @property
    def initial_distribution(self) -> np.ndarray:
        return self._pi0.copy()

    def state(self, index: int) -> DemandState:
        return self._states[index]

    def stationary_distribution(self, tol: float = 1e-12) -> np.ndarray:
        """Compute the stationary distribution via the dominant left eigenvector.

        Falls back to power iteration if the eigen-decomposition produces a
        numerically awkward answer (e.g. for near-periodic chains).
        """
        eigvals, eigvecs = np.linalg.eig(self._P.T)
        idx = np.argmin(np.abs(eigvals - 1.0))
        vec = np.real(eigvecs[:, idx])
        if vec.sum() == 0:
            vec = np.ones(self.n_states)
        vec = np.abs(vec)
        pi = vec / vec.sum()

        if not np.allclose(pi @ self._P, pi, atol=1e-6) or np.any(pi < -tol):
            pi = np.ones(self.n_states) / self.n_states
            for _ in range(10_000):
                pi_new = pi @ self._P
                if np.max(np.abs(pi_new - pi)) < tol:
                    pi = pi_new
                    break
                pi = pi_new
        return np.clip(pi, 0.0, 1.0) / np.clip(pi, 0.0, 1.0).sum()

    def distribution_at(self, t: int) -> np.ndarray:
        """State distribution after ``t`` steps starting from ``initial_distribution``."""
        if t < 0:
            raise ValueError(f"t must be >= 0, got {t}")
        pi = self._pi0
        P = self._P
        for _ in range(t):
            pi = pi @ P
        return pi

    def sample(
        self,
        n_steps: int,
        rng: np.random.Generator | None = None,
        start_state: int | None = None,
    ) -> np.ndarray:
        """Simulate a path of state indices of length ``n_steps + 1``."""
        if n_steps < 0:
            raise ValueError(f"n_steps must be >= 0, got {n_steps}")
        rng = rng if rng is not None else np.random.default_rng()
        path = np.empty(n_steps + 1, dtype=np.int64)
        if start_state is None:
            path[0] = rng.choice(self.n_states, p=self._pi0)
        else:
            path[0] = int(start_state)
        for t in range(n_steps):
            path[t + 1] = rng.choice(self.n_states, p=self._P[path[t]])
        return path


def three_state_chain(
    low: float = 80.0,
    mid: float = 100.0,
    high: float = 120.0,
    persistence: float = 0.6,
) -> MarkovChain:
    """Convenience factory for a symmetric low/mid/high demand chain.

    The transition matrix places ``persistence`` on the diagonal and
    distributes the remainder uniformly over the off-diagonal states.
    """
    if not (0.0 < persistence < 1.0):
        raise ValueError(f"persistence must be in (0, 1), got {persistence}")
    off = (1.0 - persistence) / 2.0
    P = np.array(
        [
            [persistence, off, off],
            [off, persistence, off],
            [off, off, persistence],
        ]
    )
    states = [
        DemandState(name="low", intercept=low),
        DemandState(name="mid", intercept=mid),
        DemandState(name="high", intercept=high),
    ]
    return MarkovChain(states=states, transition_matrix=P)
