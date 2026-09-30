"""Run the full comparison: 6 schedulers x 4 scenarios x 30 seeds. Writes results.json."""
import json, numpy as np, time
from ewscan import *

t0 = time.time()
w_learned = train_es(W_TEMPORAL, iters=25, pop=10, seed=0)
print("learned weights", np.round(w_learned, 4), f"({time.time()-t0:.0f}s)")

SCHED = {
    "Random":              lambda s: RandomScan(s),
    "Conventional sweep":  lambda s: Sweep(),
    "Adaptive belief":     lambda s: Belief(W_ADAPTIVE, False, False, seed=s),
    "Adaptive + temporal": lambda s: Belief(W_TEMPORAL, False, True, seed=s),
    "Learned (ES)":        lambda s: Belief(w_learned, False, True, seed=s),
    "Oracle (truth)":      lambda s: Oracle(),
}
METRICS = ["pd", "pfa", "sensitivity", "window_ratio", "emitter_ratio", "intercept_rate",
           "intercept_time", "reward", "pred_correct", "pred_err"]
SEEDS = 30
out = dict(meta=dict(bands=N_BANDS, slots=3000, seeds=SEEDS, pfa=PFA, retune_pd=RETUNE_PD,
                                learned_w=[round(float(x), 4) for x in w_learned]), scenarios={})
for sc, spec in SCENARIOS.items():
    acc = {n: {m: [] for m in METRICS} for n in SCHED}
    for sd in range(SEEDS):
        env = Env(sc, sd)
        for n, mk in SCHED.items():
            s = mk(sd); m = evaluate(env, *run(env, s, sd), getattr(s, "preds", ()))
            for k in METRICS: acc[n][k].append(m[k])
    res = {n: {k: [float(np.nanmean(v)) if not np.all(np.isnan(v)) else None,
                   float(1.96 * np.nanstd(v) / np.sqrt(np.sum(~np.isnan(v)))) if not np.all(np.isnan(v)) else None]
               for k, v in d.items()} for n, d in acc.items()}
    # demo run (first 400 slots of seed 7)
    env, demo = Env(sc, 7), {}
    for n, mk in SCHED.items():
        b, h = run(env, mk(7), 7); demo[n] = dict(b=b[:400].tolist(), h=np.nonzero(h[:400])[0].tolist())
    wins = [[int(a), int(min(e, 400)), int(b)] for a, e, b in zip(env.ws, env.we, env.wb) if a < 400]
    out["scenarios"][sc] = dict(spec=spec, metrics=res, demo=dict(windows=wins, runs=demo))
    print(sc, {n: round(res[n]["reward"][0], 2) for n in SCHED}, f"({time.time()-t0:.0f}s)")
json.dump(out, open("results.json", "w"))
