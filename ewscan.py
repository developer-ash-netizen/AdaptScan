"""ewscan: RF environment simulator + smart scan schedulers for an Electronic Support receiver.

Receiver model: N frequency bands, ONE band observed per time slot (narrow instantaneous
bandwidth). Retuning to a non-adjacent band costs detection probability (settling).
Ground truth: every transmission is a 'window' (start, end, band, emitter) in a (T x N) grid.
"""
import numpy as np

N_BANDS = 48
PFA = 0.01          # false-alarm probability per dwell on an inactive band
RETUNE_PD = 0.9     # detection-probability multiplier after a non-adjacent retune (settling loss)

SCENARIOS = {
    "mixed":    dict(cw=3,  bursty=6, periodic=4,  agile=2),
    "periodic": dict(cw=2,  bursty=2, periodic=10, agile=0),
    "agile":    dict(cw=2,  bursty=3, periodic=2,  agile=6),
    "decoys":   dict(cw=14, bursty=3, periodic=2,  agile=2),
}
WEIGHT = dict(cw=0.1, bursty=1.0, periodic=1.5, agile=2.0)   # threat weight per emitter type


# ----------------------------------------------------------------------------- environment
class Env:
    def __init__(self, scn, seed, T=3000, N=N_BANDS):
        rng = np.random.default_rng(seed)
        self.N, self.T, self.scn = N, T, scn
        W, E = [], []
        free = [int(x) for x in rng.permutation(N)]

        def add(s0, e0, b, e):
            if s0 < T:
                W.append((s0, min(e0, T), b, e))

        for kind, cnt in SCENARIOS[scn].items():
            for _ in range(cnt):
                eid, birth = len(E), int(rng.integers(0, int(0.7 * T)))
                pd = 0.95 if kind == "cw" else float(rng.uniform(0.7, 0.97))
                E.append(dict(kind=kind, w=WEIGHT[kind], pd=pd, birth=birth))
                if kind == "cw":
                    add(birth, T, free.pop(), eid)
                elif kind == "bursty":       # random bursts, geometric gaps
                    b, t = free.pop(), birth
                    while t < T:
                        L = int(rng.integers(3, 10)); add(t, t + L, b, eid)
                        t += L + 5 + int(rng.geometric(1 / 55))
                elif kind == "periodic":     # rotating-beam radar: fixed period, narrow illumination
                    b, t, P, wd = free.pop(), birth, int(rng.integers(35, 111)), int(rng.integers(2, 6))
                    E[-1]["P"] = P
                    while t < T:
                        add(t, t + wd, b, eid); t += P + int(rng.integers(-1, 2))
                else:                        # frequency-agile: random hops over a hidden hop set
                    hs = [int(x) for x in rng.choice(N, int(rng.integers(5, 10)), replace=False)]
                    D, t, pb = int(rng.integers(5, 13)), birth, -1
                    while t < T:
                        b = int(rng.choice(hs))
                        while b == pb:
                            b = int(rng.choice(hs))
                        add(t, t + D, b, eid); pb = b
                        t += D + int(rng.integers(0, 11))
        self.ws = np.array([w[0] for w in W]); self.we = np.array([w[1] for w in W])
        self.wb = np.array([w[2] for w in W]); self.wemit = np.array([w[3] for w in W])
        self.pd = np.array([e["pd"] for e in E]); self.weight = np.array([e["w"] for e in E])
        self.birth = np.array([e["birth"] for e in E]); self.kind = [e["kind"] for e in E]
        self.win_grid = -np.ones((T, N), np.int32)
        for i, (s0, e0, b, _) in enumerate(W):
            self.win_grid[s0:e0, b] = i
        order = np.argsort(self.ws, kind="stable")
        self.bw = [order[self.wb[order] == b] for b in range(N)]      # windows per band, by start
        self.bws = [self.ws[ix] for ix in self.bw]


# ----------------------------------------------------------------------------- schedulers
class Sweep:                                    # conventional open-loop sweep
    def reset(self, env): self.N = env.N
    def select(self, t): return t % self.N
    def update(self, t, b, h): pass


class RandomScan:
    def __init__(self, seed=0): self.rng = np.random.default_rng(seed)
    def reset(self, env): self.N = env.N
    def select(self, t): return int(self.rng.integers(self.N))
    def update(self, t, b, h): pass


class Oracle:                                   # sees ground truth: upper reference
    def reset(self, env): self.env, self.done_w, self.done_e = env, set(), set()
    def select(self, t):
        e = self.env; row = e.win_grid[t]; best, bb = 0.0, t % e.N
        for b in np.nonzero(row >= 0)[0]:
            w = row[b]
            if w in self.done_w: continue
            em = e.wemit[w]
            sc = e.weight[em] + (3 * e.weight[em] if em not in self.done_e else 0) + 1e-3
            if sc > best: best, bb = sc, int(b)
        return bb
    def update(self, t, b, h):
        w = self.env.win_grid[t, b]
        if h and w >= 0:
            self.done_w.add(w); self.done_e.add(self.env.wemit[w])


# feature order: belief, staleness, periodic-rendezvous, hop-transition, retune-distance, persistence
W_ADAPTIVE = np.array([1.0, 0.005, 0.0, 0.0, -0.02, 0.0])
W_TEMPORAL = np.array([1.0, 0.005, 0.03, 0.03, -0.02, 0.0])


class Belief:
    """Adaptive probability scheduler. Per-band discounted Beta belief that a dwell yields a NEW
    (informative) hit; Thompson sampling picks the band. temporal=True adds period tracking
    (rendezvous with predicted illumination windows) and a hop-transition model."""
    def __init__(self, w, thompson=True, temporal=False, gamma=0.995, a0=0.5, b0=8.0, seed=0, Lc=20.0):
        self.w, self.ts, self.temporal, self.g, self.a0, self.b0, self.Lc = np.asarray(w, float), thompson, temporal, gamma, a0, b0, Lc
        self.rng = np.random.default_rng(seed)

    def reset(self, env):
        N = self.N = env.N
        self.al, self.be = np.full(N, self.a0), np.full(N, self.b0)
        self.age, self.ar = np.zeros(N), np.arange(N)
        self.prev, self.prev_hit = -1, False
        self.lh = np.zeros(N, bool)   # was the last observation of each band a hit?
        self.ons = [[] for _ in range(N)]
        self.per, self.perlast, self.perconf = np.zeros(N), np.zeros(N), np.zeros(N)
        self.Tm, self.gl, self.preds = np.zeros((N, N)), None, []

    def select(self, t):
        N = self.N
        th = self.rng.beta(self.al, self.be) if self.ts else self.al / (self.al + self.be)
        F = np.zeros((6, N)); F[0] = th * np.minimum(self.age / self.Lc, 1.0); F[1] = np.minimum(self.age / N, 2.0)
        if self.temporal:
            has = self.per > 0
            k = np.maximum(1, np.round((t - self.perlast) / np.where(has, self.per, 1)))
            F[2] = has * self.perconf * (np.abs(t - (self.perlast + k * self.per)) <= 4)
            if self.gl and t - self.gl[0] <= 15:
                row = self.Tm[self.gl[1]]
                F[3] = row / (row.sum() + 2) * (1 - (t - self.gl[0]) / 16)
        if self.prev >= 0:
            F[4] = np.abs(self.ar - self.prev) > 1
            if self.prev_hit: F[5, self.prev] = 1.0
        return int(np.argmax(self.w @ F + self.rng.random(N) * 1e-3))

    def update(self, t, b, h):
        self.al = self.a0 + self.g * (self.al - self.a0); self.be = self.b0 + self.g * (self.be - self.b0)
        self.age += 1; self.age[b] = 0
        cont = h and bool(self.lh[b])
        if h:
            if not cont: self.al[b] += 1.0      # continuing hits (e.g. CW decoys) carry no new information
        else: self.be[b] += 1.0
        if h and not cont and self.temporal: self._onset(t, b)
        self.lh[b] = h
        self.prev, self.prev_hit = b, h

    def _onset(self, t, b):
        lst = self.ons[b]
        if lst and t - lst[-1] < 8: return
        lst.append(t); del lst[:-10]
        if self.gl and t - self.gl[0] <= 15 and self.gl[1] != b: self.Tm[self.gl[1], b] += 1
        self.gl = (t, b)
        if len(lst) >= 3:
            d = np.diff(lst); dm = d.min(); m = np.maximum(np.round(d / dm), 1)
            P = float(np.median(d / m)); conf = float(np.mean(np.abs(d - m * P) <= 4))
            if conf >= 0.75 and P >= 15:
                self.per[b], self.perlast[b], self.perconf[b] = P, t, conf
                self.preds.append((b, t, t + P))
            else:
                self.per[b] = 0


# ----------------------------------------------------------------------------- simulation + metrics
def run(env, sched, seed=0):
    rng = np.random.default_rng(seed + 99); T = env.T
    bands, hit, prev = np.zeros(T, np.int16), np.zeros(T, bool), -1
    sched.reset(env)
    for t in range(T):
        b = sched.select(t); w = env.win_grid[t, b]
        p = env.pd[env.wemit[w]] * (RETUNE_PD if prev >= 0 and abs(b - prev) > 1 else 1.0) if w >= 0 else PFA
        h = bool(rng.random() < p)
        sched.update(t, b, h); bands[t], hit[t], prev = b, h, b
    return bands, hit


def evaluate(env, bands, hit, preds=()):
    T = env.T; w = env.win_grid[np.arange(T), bands]; act = w >= 0; th = hit & act
    tt = np.nonzero(th)[0]; uw, ix = np.unique(w[th], return_index=True); ft = tt[ix]
    nE = len(env.pd); first = np.full(nE, np.inf)
    if len(uw): np.minimum.at(first, env.wemit[uw], ft)
    got = np.isfinite(first); tint = np.where(got, first - env.birth, T - env.birth)
    threat = np.array([k != "cw" for k in env.kind])
    weak_w = env.pd[env.wemit] < 0.8; hw = np.zeros(len(env.ws), bool); hw[uw] = True
    fa = int((hit & ~act).sum())
    reward = (env.weight[env.wemit[uw]].sum() + 3 * env.weight[got].sum() - 0.2 * fa) / T * 100
    err, ok = [], []
    for b, t, pt in preds:
        ww = env.win_grid[t, b]
        if ww < 0: continue
        j = np.searchsorted(env.bws[b], env.we[ww])
        if j >= len(env.bw[b]): continue
        n = env.bw[b][j]; e = abs(pt - (env.ws[n] + env.we[n]) / 2)
        err.append(e); ok.append(e <= (env.we[n] - env.ws[n]) + 2)
    return dict(
        pd=th.sum() / max(act.sum(), 1), pfa=fa / max((~act).sum(), 1),
        sensitivity=hw[weak_w].mean() if weak_w.any() else np.nan,
        window_ratio=len(uw) / len(env.ws), emitter_ratio=got.mean(),
        intercept_rate=len(uw) / T * 100, intercept_time=tint[threat].mean(), reward=reward,
        pred_correct=100 * np.mean(ok) if ok else np.nan, pred_err=np.mean(err) if err else np.nan)


def train_es(w0, iters=14, pop=8, sigma=0.3, lr=0.3, T=1200, seed=0):
    """Evolution-strategies training of the linear scoring policy on hits/misses (reward).
    Parameters are searched in a per-feature normalised space so all six weights move sensibly."""
    rng = np.random.default_rng(seed); sc = np.array([1.0, 0.01, 0.05, 0.05, 0.02, 0.02])
    th = w0 / sc
    def fit(t_, eps):
        return np.mean([evaluate(e, *run(e, Belief(t_ * sc, False, True, seed=1), 1))["reward"] for e in eps])
    for it in range(iters):
        eps = [Env(s_, 5000 + it * 10 + i, T) for i, s_ in enumerate(SCENARIOS) for i in range(2)]
        nz = rng.standard_normal((pop, len(th))); nz[:, 0] = 0
        f = np.array([fit(th + sigma * n, eps) for n in nz])
        th = th + lr * ((f - f.mean()) / (f.std() + 1e-9) @ nz) / pop * 2
    val = [Env(s_, 9000 + i, T) for i, s_ in enumerate(SCENARIOS) for i in range(3)]
    return th * sc if fit(th, val) > fit(w0 / sc, val) else w0
