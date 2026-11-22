from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from scheduling import assign_users


@dataclass
class EnvConfig:
    N: int = 12
    C: tuple = (4.0, 6.0, 10.0)
    eta: float = 2.0
    B: int = 4
    kappa: float = 2.0
    lam: float = 1.0
    tau: tuple = (0.75, 1.25)
    T: int = 50
    gamma: float = 0.95
    user_policy: str = "selfish"
    load_low: float = 1.2
    load_high: float = 2.2
    p_switch: float = 0.1
    backlog_cap: float = 1.0

    @property
    def K(self) -> int:
        return len(self.C)


def action_space(K: int, B: int) -> np.ndarray:
    acts = [d for d in itertools.product(range(B + 1), repeat=K) if sum(d) <= B]
    return np.array(acts, dtype=int)


def state_index(x: np.ndarray) -> int:
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

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.backlog = np.zeros(self.K)
        self.regime = int(self.rng.random() < 0.5)
        self.prev_d = np.zeros(self.K, dtype=int)
        self.assign = self.rng.integers(0, self.K, size=self.cfg.N)
        self._arrivals()
        return self.observe()

    def _arrivals(self):
        cfg = self.cfg
        if self.rng.random() < cfg.p_switch:
            self.regime = 1 - self.regime
        mean = cfg.load_high if self.regime else cfg.load_low
        self.L = mean * self.rng.uniform(0.5, 1.5, size=cfg.N)
        cap_seen = self.C + cfg.eta * self.prev_d
        self.assign = assign_users(cfg.user_policy, self.L, self.backlog, cap_seen,
                                   cfg.kappa, self.rng, init=self.assign)

    def loads(self) -> np.ndarray:
        W = np.bincount(self.assign, weights=self.L, minlength=self.K)
        return self.backlog + W

    def observe(self) -> np.ndarray:
        z = self.loads() / self.C
        t1, t2 = self.cfg.tau
        return np.digitize(z, [t1, t2])

    def delays(self, d: np.ndarray) -> np.ndarray:
        cap = self.C + self.cfg.eta * d
        Dj = (self.loads() / cap) ** self.cfg.kappa
        return Dj[self.assign]

    def reward_parts(self, d: np.ndarray):
        total_delay = float(self.delays(d).sum())
        cost = self.cfg.lam * float(np.sum(d))
        return -total_delay - cost, total_delay

    def step(self, a_idx: int):
        d = self.actions[a_idx]
        r, total_delay = self.reward_parts(d)
        cap = self.C + self.cfg.eta * d
        self.backlog = np.minimum(np.maximum(self.loads() - cap, 0.0),
                                  self.cfg.backlog_cap * self.C)
        self.prev_d = d.copy()
        self.t += 1
        done = self.t >= self.cfg.T
        self._arrivals()
        info = {"delay": total_delay, "units": int(d.sum())}
        return self.observe(), r, done, info
