"""Congestion environment for multi-agent compute scheduling.

Implements the MDP M = (S, D, P, R, gamma) described in the report:
  * N users submit one job each step to one of K heterogeneous servers.
  * A resource manager allocates up to B extra capacity units (each worth eta).
  * State  x in {0,1,2}^K : discretised normalised load of every server.
  * Action d in D = {d in Z^K_{>=0} : sum d <= B}.
  * Reward R = - sum_i D_i(a, d) - lambda * sum_j d_j,
           D_j = ((b_j + W_j) / (C_j + eta d_j))^kappa.
  * Transition: unfinished work (backlog) carries over, users re-select servers
    with the scheduling rule (random / selfish / cooperative) after observing the
    previous allocation, and workloads follow a hidden two-regime demand process.
  * Episode terminates after T decision steps.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from scheduling import assign_users


@dataclass
class EnvConfig:
    N: int = 12                                # number of users
    C: tuple = (4.0, 6.0, 10.0)                 # base capacity of each server
    eta: float = 2.0                            # capacity added per resource unit
    B: int = 4                                  # resource budget per step
    kappa: float = 2.0                          # delay exponent
    lam: float = 1.0                            # resource cost lambda
    tau: tuple = (0.75, 1.25)                   # discretisation thresholds tau1 < tau2
    T: int = 50                                 # episode horizon
    gamma: float = 0.95                         # discount factor
    user_policy: str = "selfish"                # random | selfish | cooperative
    load_low: float = 1.2                       # mean job size, low-demand regime
    load_high: float = 2.2                      # mean job size, high-demand regime
    p_switch: float = 0.1                       # regime switching probability
    backlog_cap: float = 1.0                    # max carried backlog, as multiple of C_j

    @property
    def K(self) -> int:
        return len(self.C)


def action_space(K: int, B: int) -> np.ndarray:
    """All integer vectors d >= 0 with sum(d) <= B (Section 3 of the report)."""
    acts = [d for d in itertools.product(range(B + 1), repeat=K) if sum(d) <= B]
    return np.array(acts, dtype=int)


def state_index(x: np.ndarray) -> int:
    """Map x in {0,1,2}^K to an integer in [0, 3^K)."""
    idx = 0
    for v in x:
        idx = idx * 3 + int(v)
    return idx


def index_state(idx: int, K: int) -> np.ndarray:
    x = np.zeros(K, dtype=int)
    for j in range(K - 1, -1, -1):
        x[j] = idx % 3
        idx //= 3
    return x


class CongestionEnv:
    def __init__(self, cfg: EnvConfig, seed: int | None = None):
        self.cfg = cfg
        self.C = np.asarray(cfg.C, dtype=float)
        self.K = cfg.K
        self.actions = action_space(self.K, cfg.B)
        self.nS = 3 ** self.K
        self.nA = len(self.actions)
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ core
    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.backlog = np.zeros(self.K)
        self.regime = int(self.rng.random() < 0.5)       # 0 = low, 1 = high demand
        self.prev_d = np.zeros(self.K, dtype=int)
        self.assign = self.rng.integers(0, self.K, size=self.cfg.N)
        self._arrivals()
        return self.observe()

    def _arrivals(self):
        """Draw new workloads and let users choose servers (step 5 of Sec 5.1)."""
        cfg = self.cfg
        if self.rng.random() < cfg.p_switch:
            self.regime = 1 - self.regime
        mean = cfg.load_high if self.regime else cfg.load_low
        self.L = mean * self.rng.uniform(0.5, 1.5, size=cfg.N)
        # users only know the capacities produced by the *previous* allocation
        cap_seen = self.C + cfg.eta * self.prev_d
        self.assign = assign_users(cfg.user_policy, self.L, self.backlog, cap_seen,
                                   cfg.kappa, self.rng, init=self.assign)

    def loads(self) -> np.ndarray:
        """Total work at each server: carried backlog + newly assigned workload."""
        W = np.bincount(self.assign, weights=self.L, minlength=self.K)
        return self.backlog + W

    def observe(self) -> np.ndarray:
        z = self.loads() / self.C                      # normalised load z_j
        t1, t2 = self.cfg.tau
        return np.digitize(z, [t1, t2])                # 0 / 1 / 2 congestion level

    def delays(self, d: np.ndarray) -> np.ndarray:
        cap = self.C + self.cfg.eta * d
        Dj = (self.loads() / cap) ** self.cfg.kappa
        return Dj[self.assign]                         # delay of every user

    def reward_parts(self, d: np.ndarray):
        total_delay = float(self.delays(d).sum())
        cost = self.cfg.lam * float(np.sum(d))
        return -total_delay - cost, total_delay

    def step(self, a_idx: int):
        d = self.actions[a_idx]
        r, total_delay = self.reward_parts(d)
        # service: each server processes C_j + eta d_j units; the rest carries over
        cap = self.C + self.cfg.eta * d
        self.backlog = np.minimum(np.maximum(self.loads() - cap, 0.0),
                                  self.cfg.backlog_cap * self.C)
        self.prev_d = d.copy()
        self.t += 1
        done = self.t >= self.cfg.T
        self._arrivals()
        info = {"delay": total_delay, "units": int(d.sum())}
        return self.observe(), r, done, info
