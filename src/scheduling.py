from __future__ import annotations

import numpy as np


def _server_delay(work, cap, kappa):
    return (work / cap) ** kappa


def social_cost(assign, L, backlog, cap, kappa):
    K = len(cap)
    W = backlog + np.bincount(assign, weights=L, minlength=K)
    n = np.bincount(assign, minlength=K)
    return float(np.sum(n * _server_delay(W, cap, kappa)))


def selfish(L, backlog, cap, kappa, rng, init=None, max_rounds=50):
    N, K = len(L), len(cap)
    a = rng.integers(0, K, size=N) if init is None else init.copy()
    W = backlog + np.bincount(a, weights=L, minlength=K)
    for _ in range(max_rounds):
        changed = False
        for i in rng.permutation(N):
            W_wo = W.copy()
            W_wo[a[i]] -= L[i]
            cost = _server_delay(W_wo + L[i], cap, kappa)
            best = int(np.argmin(cost))
            if cost[best] < cost[a[i]] - 1e-12:
                W = W_wo
                W[best] += L[i]
                a[i] = best
                changed = True
        if not changed:
            break
    return a


def cooperative(L, backlog, cap, kappa, rng, init=None, max_iter=50):
    N, K = len(L), len(cap)
    a = np.zeros(N, dtype=int)
    W = backlog.astype(float).copy()
    n = np.zeros(K)
    for i in np.argsort(-L):
        base = np.sum(n * _server_delay(W, cap, kappa))
        best, best_c = 0, np.inf
        for j in range(K):
            W[j] += L[i]; n[j] += 1
            c = np.sum(n * _server_delay(W, cap, kappa)) - base
            W[j] -= L[i]; n[j] -= 1
            if c < best_c:
                best, best_c = j, c
        a[i] = best; W[best] += L[i]; n[best] += 1
    a = a.tolist(); W = W.tolist(); n = n.tolist(); c_ = list(cap); Ls = list(L)

    def sc(j, w, m):
        return m * (w / c_[j]) ** kappa

    for _ in range(max_iter):
        improved = False
        for i in range(N):
            p = a[i]
            for j in range(K):
                if j == p:
                    continue
                before = sc(p, W[p], n[p]) + sc(j, W[j], n[j])
                after = sc(p, W[p] - Ls[i], n[p] - 1) + sc(j, W[j] + Ls[i], n[j] + 1)
                if after < before - 1e-12:
                    W[p] -= Ls[i]; n[p] -= 1; W[j] += Ls[i]; n[j] += 1
                    a[i] = j; p = j; improved = True
        for i in range(N):
            for k in range(i + 1, N):
                p, q = a[i], a[k]
                if p == q:
                    continue
                dl = Ls[k] - Ls[i]
                before = sc(p, W[p], n[p]) + sc(q, W[q], n[q])
                after = sc(p, W[p] + dl, n[p]) + sc(q, W[q] - dl, n[q])
                if after < before - 1e-12:
                    W[p] += dl; W[q] -= dl
                    a[i], a[k] = q, p; improved = True
        if not improved:
            break
    return np.array(a, dtype=int)


def assign_users(policy, L, backlog, cap, kappa, rng, init=None):
    if policy == "random":
        return rng.integers(0, len(cap), size=len(L))
    if policy == "selfish":
        return selfish(L, backlog, cap, kappa, rng, init=init)
    if policy == "cooperative":
        return cooperative(L, backlog, cap, kappa, rng)
    raise ValueError(policy)
