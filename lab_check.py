"""
Локальный gate для AI-агента в песочнице лаборатории: те же семейства миров, что у gate (`lab.matrix`), но свои seed —
gate считает на отложенных, которых агент не видит (иначе цикл подгоняет agent.py под тест). Родитель (`agent_parent.py`)
прогоняется на тех же seed один раз и кэшируется в lab_check.json; сравнение попарное по мирам, как в gate.
`paired` и `verdict` — общие с `lab.gate`.

    python3 lab_check.py      # ~10 с на 8 ядрах (+10 с на первом запуске — родитель); последняя строка — вердикт
"""

import hashlib
import json
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
PRIMARY = "harsh0"
GATE_P = 0.9  # уверенность gate: «хуже» и «лучше» — только при P ≥ 90% по бутстрепу попарных разниц
# ключ метрик → (подпись, keep жёсткого мира; None — стресс-мир), как WORLDS в lab.py
FAMILIES = {"harsh0": ("жёсткие миры, история бесполезна", 0.0), "harsh50": ("жёсткие миры, история наполовину верна", 0.5),
            "stress": ("стресс-миры", None)}


def _use(path):
    """Инициализатор воркера: модуль agent из другого файла (родитель)."""
    if path:
        import importlib.util
        spec = importlib.util.spec_from_file_location("agent", path)
        m = importlib.util.module_from_spec(spec)
        sys.modules["agent"] = m
        spec.loader.exec_module(m)


def one(job):
    """Один мир — как lab.run_once: падение агента тоже результат, пилоты входят в счёт."""
    key, seed = job
    import pandas as pd
    import agent
    import stress_eval as se
    from environment import make_environment
    from mock_environment import CHANNELS, MAX_TOTAL_CONTACTS, TOTAL_BUDGET, _mock_fallback
    from scoring_core import sanitize_campaigns, score_campaigns
    keep = FAMILIES[key][1]
    model = se.world(seed) if keep is None else se.harsh_world(seed, keep)
    env, internals = make_environment(se.profile, model, se.dict_tariff, CHANNELS, TOTAL_BUDGET, MAX_TOTAL_CONTACTS,
                                      _mock_fallback, seed=seed)
    t, crash = time.time(), None
    try:
        raw = agent.Agent().act(env) or []
    except Exception as e:
        raw, crash = [], f"{type(e).__name__}: {e}"
    camps = sanitize_campaigns(raw, env.tariffs)[:10]
    df = pd.DataFrame(internals.executed_pilot_campaigns() + camps)
    for c in se.COLS:
        if c not in df.columns:
            df[c] = None
    net = score_campaigns(df, se.profile, model, se.dict_tariff, se.profile["predicted_arpu"].sum(), _mock_fallback)["net_arpu_gain"]
    return key, seed, float(net), crash, len(raw) - len(camps), time.time() - t


def _m(x):
    return f"{x / 1e6:+.2f}M"


def paired(new, old, tol=0.0, b=4000):
    """
    new / old: {seed: net} версии и родителя на одних мирах. Бутстреп среднего попарной разницы:
    delta, 95% ДИ, P(лучше) = P(Δ > 0), P(хуже) = P(Δ < −tol·|среднее родителя|).
    """
    seeds = sorted(new.keys() & old.keys())
    d = np.array([new[s] - old[s] for s in seeds])
    means = d[np.random.default_rng(0).integers(0, len(d), (b, len(d)))].mean(1)
    floor = -tol * abs(np.mean([old[s] for s in seeds]))
    return {"n": len(d), "delta": float(d.mean()), "lo": float(np.percentile(means, 2.5)), "hi": float(np.percentile(means, 97.5)),
            "p_better": float((means > 0).mean()), "p_worse": float((means < floor).mean())}


def verdict(new, old, tol, p=GATE_P):
    """
    new / old: семейство → {seed: net}. → (причины отказа, статистика по семействам); та же логика у lab.gate.
    Отказ: какое-то семейство уверенно хуже (главное — без допуска, остальные — больше чем на tol)
    или ни в одном нет уверенного улучшения: нейтральная правка — шум, а не прогресс.
    """
    reasons, stats = [], {}
    for key, (label, _) in FAMILIES.items():
        if new.get(key) and old.get(key):
            t = 0.0 if key == PRIMARY else tol
            s = stats[key] = paired(new[key], old[key], t)
            if s["p_worse"] >= p:
                reasons.append(f"«{label}» хуже{f' больше чем на {t:.0%}' if t else ''}: Δ {_m(s['delta'])} на мир "
                               f"(95% ДИ {_m(s['lo'])}…{_m(s['hi'])}), P(хуже) = {s['p_worse']:.0%}")
    if stats and max(s["p_better"] for s in stats.values()) < p:
        reasons.append("улучшение не доказано: " + ", ".join(f"{k} P(лучше) = {s['p_better']:.0%}" for k, s in stats.items())
                       + f" — нужно ≥ {p:.0%} хотя бы в одном семействе")
    return reasons, stats


def _nets(res):
    out = {}
    for key, seed, net, *_ in res:
        out.setdefault(key, {})[int(seed)] = net
    return out


def _run(jobs, workers, source=None):
    with ProcessPoolExecutor(workers, initializer=_use, initargs=(source,)) as pool:
        return list(pool.map(one, jobs))


def main():
    cfg_path = HERE / "lab_check.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    seeds, primary = cfg.get("seeds", list(range(10))), cfg.get("primary_seeds", list(range(30)))
    jobs = [(k, s) for k in FAMILIES for s in (primary if k == PRIMARY else seeds)]
    workers = cfg.get("workers") or max(2, min(8, (os.cpu_count() or 2) - 1))
    t = time.time()
    parent = HERE / "agent_parent.py"
    cache = cfg.get("parent_nets") or {}
    if parent.exists():
        sha = hashlib.sha256(parent.read_bytes()).hexdigest()
        if cache.get("sha") != sha or cache.get("jobs") != [list(j) for j in jobs]:
            print("прогоняю родителя на тех же мирах (один раз)…", flush=True)
            cache = cfg["parent_nets"] = {"sha": sha, "jobs": [list(j) for j in jobs], "nets": _nets(_run(jobs, workers, str(parent)))}
            cfg_path.write_text(json.dumps(cfg))
    old = {k: {int(s): n for s, n in d.items()} for k, d in cache.get("nets", {}).items()}
    res = _run(jobs, workers)
    new = _nets(res)
    reasons, stats = verdict(new, old, cfg.get("tol", 0.03)) if old else ([], {})
    print(f"{len(jobs)} миров за {time.time() - t:.0f} с; родитель {cfg.get('parent_id', '—')}")
    print(f"{'семейство':<10} {'медиана':>9} {'родитель':>9} {'Δ на мир':>9} {'95% ДИ':>17} {'P(лучше)':>9}")
    for key in FAMILIES:
        s, med = stats.get(key), statistics.median(new[key].values())
        pm = f"{_m(statistics.median(old[key].values()))}" if old.get(key) else "—"
        print(f"{key:<10} {_m(med):>9} {pm:>9} " + (f"{_m(s['delta']):>9} {_m(s['lo']) + '…' + _m(s['hi']):>17} {s['p_better']:>9.0%}" if s else ""))
    fail = [f"падений агента: {n} (первое: {next(r[3] for r in res if r[3])})" for n in [sum(bool(r[3]) for r in res)] if n]
    fail += [f"отброшенных кампаний: {n}" for n in [sum(r[4] for r in res)] if n]
    fail += [f"прогон дольше {cfg.get('runtime', 600)} с" for _ in [0] if max(r[5] for r in res) >= cfg.get("runtime", 600)]
    reasons = fail + reasons
    if not old:
        print("ИТОГ: нет agent_parent.py — сравнить не с чем, смотри абсолютные числа")
    else:
        print("ИТОГ: ПРОШЛА БЫ GATE" if not reasons else "ИТОГ: НЕ ПРОШЛА — " + "; ".join(reasons))


if __name__ == "__main__":
    main()
