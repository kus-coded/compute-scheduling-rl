"""Resource-manager policies.

Learning / planning agents
  * QLearningAgent  - tabular Q-learning with epsilon-greedy exploration (Sec 7).
  * DPAgent         - model-based dynamic programming: P(x'|x,d) and R(x,d) are
                      estimated from exploration data, then value iteration.
Baselines
  * NoAlloc         - never allocate extra capacity (d = 0).
  * Uniform         - always spend the full budget, spread evenly.
  * Threshold       - state-based heuristic: give units to the most congested
                      servers (level 2 first, then level 1), smallest capacity first.
  * MyopicOracle    - observes exact loads (not just the discrete state) and picks
                      the action maximising the immediate reward. Reference only.
"""
from __future__ import annotations

import numpy as np

from env import CongestionEnv, state_index


class QLearningAgent:
    def __init__(self, nS, nA, alpha=0.1, gamma=0.95, eps_start=1.0, eps_end=0.05,
                 eps_decay_frac=0.6, seed=0, omega=0.6, alpha_min=0.01):
        self.Q = np.zeros((nS, nA))
        self.alpha, self.gamma = alpha, gamma
        self.omega, self.alpha_min = omega, alpha_min
        self.eps_start, self.eps_end, self.eps_decay_frac = eps_start, eps_end, eps_decay_frac
        self.rng = np.random.default_rng(seed)
        self.visits = np.zeros((nS, nA), dtype=int)

    def epsilon(self, ep, n_ep):
        frac = min(1.0, ep / (self.eps_decay_frac * n_ep))
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def act(self, s, eps=0.0):
        if self.rng.random() < eps:
            return int(self.rng.integers(self.Q.shape[1]))
        q = self.Q[s]
        return int(self.rng.choice(np.flatnonzero(q == q.max())))

    def update(self, s, a, r, s2, done):
        target = r if done else r + self.gamma * self.Q[s2].max()
        self.visits[s, a] += 1
        if self.alpha == "visit":       # alpha_n = n(s,a)^-omega (Robbins-Monro), floored
            alpha = max(self.alpha_min, self.visits[s, a] ** -self.omega)
        else:
            alpha = self.alpha
        self.Q[s, a] = (1 - alpha) * self.Q[s, a] + alpha * target

    def train(self, env: CongestionEnv, n_episodes, seed_offset=0, log_every=1):
        returns = []
        for ep in range(n_episodes):
            x = env.reset(seed=seed_offset + ep)
            s = state_index(x)
            eps = self.epsilon(ep, n_episodes)
            G, disc, done = 0.0, 1.0, False
            while not done:
                a = self.act(s, eps)
                x2, r, done, _ = env.step(a)
                s2 = state_index(x2)
                self.update(s, a, r, s2, done)
                G += disc * r; disc *= self.gamma
                s = s2
            returns.append(G)
        return np.array(returns)

    def policy(self, x, env=None):
        return self.act(state_index(x), 0.0)


class DPAgent:
    """Certainty-equivalent DP: estimate the MDP from data, then value iteration."""

    def __init__(self, nS, nA, gamma=0.95):
        self.nS, self.nA, self.gamma = nS, nA, gamma

    def fit_model(self, env: CongestionEnv, n_episodes, seed_offset=0, seed=0):
        rng = np.random.default_rng(seed)
        counts = np.zeros((self.nS, self.nA, self.nS))
        rsum = np.zeros((self.nS, self.nA))
        for ep in range(n_episodes):
            x = env.reset(seed=seed_offset + ep)
            s, done = state_index(x), False
            while not done:
                a = int(rng.integers(self.nA))
                x2, r, done, _ = env.step(a)
                s2 = state_index(x2)
                counts[s, a, s2] += 1; rsum[s, a] += r
                s = s2
        n = counts.sum(axis=2)
        self.n_sa = n
        self.R = np.where(n > 0, rsum / np.maximum(n, 1), rsum[n > 0].min() if (n > 0).any() else 0)
        P = counts / np.maximum(n[..., None], 1)
        P[n == 0] = 1.0 / self.nS                    # unseen (s,a): uniform
        self.P = P
        return self

    def solve(self, tol=1e-8, max_iter=10000):
        V = np.zeros(self.nS)
        for it in range(max_iter):
            Qsa = self.R + self.gamma * self.P @ V
            V_new = Qsa.max(axis=1)
            if np.max(np.abs(V_new - V)) < tol:
                V = V_new
                break
            V = V_new
        self.V, self.Q, self.iters = V, self.R + self.gamma * self.P @ V, it + 1
        self.pi = self.Q.argmax(axis=1)
        return self

    def policy(self, x, env=None):
        return int(self.pi[state_index(x)])


def _action_lookup(env):
    return {tuple(d): i for i, d in enumerate(env.actions)}


class NoAlloc:
    name = "No allocation"

    def __init__(self, env):
        self.a = _action_lookup(env)[tuple([0] * env.K)]

    def policy(self, x, env=None):
        return self.a


class Uniform:
    name = "Uniform (full budget)"

    def __init__(self, env):
        K, B = env.K, env.cfg.B
        d = [B // K] * K
        for j in np.argsort(env.C)[: B - sum(d)]:     # remainder to smallest servers
            d[j] += 1
        self.a = _action_lookup(env)[tuple(d)]

    def policy(self, x, env=None):
        return self.a


class Threshold:
    name = "Threshold heuristic"

    def __init__(self, env):
        self.lookup = _action_lookup(env)
        self.K, self.B, self.C = env.K, env.cfg.B, env.C

    def policy(self, x, env=None):
        d = np.zeros(self.K, dtype=int)
        budget = self.B
        # level-2 servers get 2 units, then level-1 servers get 1 unit
        for lvl, units in ((2, 2), (1, 1)):
            for j in sorted(np.flatnonzero(x == lvl), key=lambda j: self.C[j]):
                give = min(units, budget)
                d[j] += give; budget -= give
        return self.lookup[tuple(d)]


class MyopicOracle:
    name = "Myopic oracle (exact loads)"

    def policy(self, x, env):
        best, best_r = 0, -np.inf
        for i, d in enumerate(env.actions):
            r, _ = env.reward_parts(d)
            if r > best_r:
                best, best_r = i, r
        return best


def evaluate(env: CongestionEnv, agent, n_episodes=200, seed_offset=10_000_000):
    """Run greedy policy with common random numbers; return per-episode metrics."""
    G_all, delay_all, units_all = [], [], []
    for ep in range(n_episodes):
        x = env.reset(seed=seed_offset + ep)
        done, G, disc, dl, un = False, 0.0, 1.0, 0.0, 0
        while not done:
            a = agent.policy(x, env)
            x, r, done, info = env.step(a)
            G += disc * r; disc *= env.cfg.gamma
            dl += info["delay"]; un += info["units"]
        G_all.append(G); delay_all.append(dl / env.cfg.T); units_all.append(un / env.cfg.T)
    return {"return": np.array(G_all), "delay": np.array(delay_all), "units": np.array(units_all)}
