"""Round 12: exact modulo scheduling with register pressure, by CP-SAT (OR-tools), for r11/msched.py loops.

msched's list / iterative schedulers place ops as early as resources allow and only then shorten lifetimes (sink pass);
at a tight II that leaves the register file over-subscribed (the GELU loop at II 10: MaxLive 12-13 of 8 x registers).
Here the whole schedule is one constraint model:
  - every op gets one issue cycle t in [0, H) (one-hot booleans), t_succ >= t_pred + distance for msched's edges;
  - each slot / itinerary port / bank is used at most once per cycle modulo II (Loop._res: (stage, resource) pairs);
  - register pressure, exactly as msched counts it: vreg v is live at iteration-relative cycle c iff
    lo_v <= c <= hi_v (lo = earliest def landing t + lat, hi = last read t + stage + slack); the steady state holds
    sum_{c = m mod II} live_v(c) copies at residue m, and that sum over a class's vregs must fit the class's registers
    (minus `reserve`, kept free so circular-arc allocation over U * II cycles has room);
  - max_life: every register lifetime <= max_life cycles (bounds the modulo-variable-expansion unroll U);
  - last_stage [ops]: placed in the schedule's last stage (side effects never run in the drain's garbage iterations);
  - maxdist [(a, b, d)]: t_b - t_a <= d (with an `after` edge a -> b: a stream's ops of one iteration stay in order inside
    one II, so its post-modified pointer walks in iteration order).
Objective (lexicographic by weights): the x / dm MaxLive margin, the schedule length (stages: prologue + drain cycles
per call), the summed lifetimes. The result is a {op name: t} map; msched then allocates (modulo variable expansion),
emits and checks it as usual (`Loop.place(times)`).

The solver runs on the laptop (`uv run --no-project --with ortools --with numpy python r12/...`); generators save the
times as JSON next to themselves, so builds on testbox (npu-iron, no OR-tools) replay a fixed schedule.
"""

from collections import defaultdict


def solve(lp, II, H=None, budgets=None, reserve=None, slack=0, max_len=None, time_limit=60.0, workers=8, verbose=False,
          w_len=1, w_life=0, fix=None, maxdist=(), last_stage=(), max_life=None, min_t=None, hint=None):
    """lp: an msched.Loop (ops with .slot / .ports / defs / uses, lp.preds edges). budgets: class -> registers usable
    (default: len(lp.classes[c])). reserve: class -> registers kept free in the pressure bound. Returns {op: t} or None."""
    from ortools.sat.python import cp_model

    ops = lp.ops
    H = H or 8 * II
    budgets = dict(budgets or {c: len(r) for c, r in lp.classes.items()})
    reserve = reserve or {}
    m = cp_model.CpModel()
    x = {(o.name, t): m.NewBoolVar(f"x_{o.name}_{t}") for o in ops for t in range(H)}
    T = {}
    for o in ops:
        m.AddExactlyOne(x[o.name, t] for t in range(H))
        T[o.name] = m.NewIntVar(0, H - 1, f"t_{o.name}")
        m.Add(T[o.name] == sum(t * x[o.name, t] for t in range(H)))
        if fix and o.name in fix:
            m.Add(T[o.name] == fix[o.name])
    for a, b, d in maxdist:  # t_b - t_a <= d (e.g. a stream's loads of one iteration inside one II, in order)
        m.Add(T[b] - T[a] <= d)
    for n, v in (min_t or {}).items():  # round 13: an op no earlier than v (e.g. a store out of the first stage)
        m.Add(T[n] >= v)
    for n, v in (hint or {}).items():  # round 13: a warm start (e.g. msched's schedule)
        m.AddHint(T[n], v)
    for n, ps in lp.preds.items():  # dependences (all intra-iteration in these loops)
        for p, d in ps:
            m.Add(T[n] >= T[p] + d)
    for n, ps in getattr(lp, "cpreds", {}).items():  # round 16: loop-carried (the next iteration reads): t_n + II >= t_p + d
        for p, d in ps:
            m.Add(T[n] + II >= T[p] + d)
    # resources modulo II
    use = defaultdict(list)
    for o in ops:
        for st, k in lp._res(o):
            for t in range(H):
                use[k, (t + st) % II].append(x[o.name, t])
    for (k, r), vs in use.items():
        if len(vs) > 1:
            m.Add(sum(vs) <= 1)
    # register pressure
    defs, uses = defaultdict(list), defaultdict(list)
    for o in ops:
        for v, p, lat in o.defs:
            defs[v].append((o.name, lat))
        for v, p, st in o.uses:
            uses[v].append((o.name, st))
        for v, p, st in getattr(o, "cuses", ()):  # round 16: carried reads happen one II later
            uses[v].append((o.name, st + II))
    C = H + 16  # cycles a lifetime may reach (last read of the last op)
    prefix = {}  # (op, c) -> [t_op <= c] as a linear expression (sum of one-hot booleans)

    def le(n, c):  # [t_n <= c]
        c = min(c, H - 1)
        if c < 0:
            return 0
        if (n, c) not in prefix:
            prefix[n, c] = sum(x[n, t] for t in range(c + 1))
        return prefix[n, c]

    live = defaultdict(dict)
    lo_v, hi_v = {}, {}
    for v in defs:
        cls = lp.vclass[v]
        lo_v[v] = m.NewIntVar(0, C, f"lo_{v}")
        hi_v[v] = m.NewIntVar(0, C, f"hi_{v}")
        m.AddMinEquality(lo_v[v], [T[n] + lat for n, lat in defs[v]])
        if uses[v]:
            m.AddMaxEquality(hi_v[v], [T[n] + st + slack for n, st in uses[v]])
        else:
            m.Add(hi_v[v] == lo_v[v])
        if max_life is not None:  # every lifetime within max_life cycles (U copies at II: max_life = U * II)
            m.Add(hi_v[v] - lo_v[v] + 1 <= max_life)
        for c in range(C):
            a = m.NewBoolVar(f"a_{v}_{c}")  # >= [lo_v <= c]: some def landed by c
            for n, lat in defs[v]:
                m.Add(a >= le(n, c - lat))
            b = m.NewBoolVar(f"b_{v}_{c}")  # >= [hi_v >= c]: some read at or after c
            for n, st in uses[v]:
                m.Add(b >= 1 - le(n, c - st - slack - 1))
            lv = m.NewBoolVar(f"l_{v}_{c}")
            m.Add(lv >= a + b - 1)
            live[cls][v, c] = lv
    peak = {}
    for cls, lvs in live.items():
        peak[cls] = m.NewIntVar(0, 64, f"peak_{cls}")
        for r in range(II):
            m.Add(sum(lv for (v, c), lv in lvs.items() if c % II == r) <= peak[cls])
        m.Add(peak[cls] <= budgets[cls] - reserve.get(cls, 0))
    first = m.NewIntVar(0, H - 1, "first")  # the schedule starts at 0 (stage arithmetic below; shifting is free otherwise)
    m.AddMinEquality(first, [T[o.name] for o in ops])
    m.Add(first == 0)
    length = m.NewIntVar(0, C, "length")
    m.AddMaxEquality(length, [T[o.name] for o in ops])
    if max_len is not None:
        m.Add(length <= max_len)
    if last_stage:  # ops with side effects (pushes, flushes) in the final stage: the drain's garbage iterations never run them
        k = m.NewIntVar(0, C, "last_stage")
        m.AddDivisionEquality(k, length, II)
        for n in last_stage:
            m.Add(T[n] >= k * II)
    life = sum(hi_v[v] - lo_v[v] for v in defs)
    m.Minimize(100 * sum(peak.values()) + w_len * length + w_life * life)
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = time_limit
    s.parameters.num_workers = workers
    s.parameters.log_search_progress = verbose
    st = s.Solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    out = {o.name: int(s.Value(T[o.name])) for o in ops}
    t0 = min(out.values())
    info = dict(status=s.StatusName(st), peak={c: int(s.Value(p)) for c, p in peak.items()}, length=int(s.Value(length)) - t0)
    return {n: t - t0 for n, t in out.items()}, info
