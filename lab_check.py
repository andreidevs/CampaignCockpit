"""
Локальный gate для AI-агента в песочнице лаборатории: те же семейства миров и seed, что у gate
(`lab.matrix`), на текущем agent.py — и сравнение с метриками родителя из lab_check.json (кладёт lab._sandbox).
Агенту не нужен.

    python3 lab_check.py      # ~10 с на 8 ядрах; последняя строка — «ИТОГ: ПРОШЛА БЫ GATE» или причины отказа
"""

import json
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
PRIMARY = "harsh0"
# ключ метрик → (подпись, keep жёсткого мира; None — стресс-мир), как WORLDS в lab.py
FAMILIES = {"harsh0": ("жёсткие миры, история бесполезна", 0.0), "harsh50": ("жёсткие миры, история наполовину верна", 0.5),
            "stress": ("стресс-миры", None)}


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


def verdict(m, pm, tol):
    """Причины отказа gate по метрикам (та же логика, что lab.gate; сверяется в self-check lab.py)."""
    reasons = []
    for key, (label, _) in FAMILIES.items():
        if key not in pm or key not in m:
            continue
        a, b = pm[key]["median"], m[key]["median"]
        t = 0.0 if key == PRIMARY else tol
        if b < a - t * abs(a):
            reasons.append(f"медиана «{label}» упала: {_m(a)} → {_m(b)}" + (f" (допуск {t:.0%})" if t else ""))
    if PRIMARY in pm and PRIMARY in m and m[PRIMARY]["min"] < pm[PRIMARY]["min"]:
        reasons.append(f"худший мир «{FAMILIES[PRIMARY][0]}» стал хуже: {_m(pm[PRIMARY]['min'])} → {_m(m[PRIMARY]['min'])}")
    return reasons


def main():
    cfg_path = HERE / "lab_check.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    seeds, primary = cfg.get("seeds", list(range(10))), cfg.get("primary_seeds", list(range(30)))
    pm = cfg.get("parent") or {}
    jobs = [(k, s) for k in FAMILIES for s in (primary if k == PRIMARY else seeds)]
    t = time.time()
    with ProcessPoolExecutor(cfg.get("workers") or max(2, min(8, (os.cpu_count() or 2) - 1))) as pool:
        res = list(pool.map(one, jobs))
    m = {}
    for key in FAMILIES:
        nets = [r[2] for r in res if r[0] == key]
        m[key] = {"median": statistics.median(nets), "min": min(nets), "n": len(nets)}
    print(f"{len(jobs)} миров за {time.time() - t:.0f} с; родитель {cfg.get('parent_id', '—')}")
    print(f"{'семейство':<10} {'медиана':>9} {'мин':>9}   {'родитель: медиана':>18} {'мин':>9}")
    for key in FAMILIES:
        p = pm.get(key) or {}
        print(f"{key:<10} {_m(m[key]['median']):>9} {_m(m[key]['min']):>9}   "
              f"{_m(p['median']) if p.get('median') is not None else '—':>18} {_m(p['min']) if p.get('min') is not None else '—':>9}")
    reasons = [f"падений агента: {n} (первое: {next(r[3] for r in res if r[3])})" for n in [sum(bool(r[3]) for r in res)] if n]
    reasons += [f"отброшенных кампаний: {n}" for n in [sum(r[4] for r in res)] if n]
    reasons += [f"прогон дольше {cfg.get('runtime', 600)} с" for _ in [0] if max(r[5] for r in res) >= cfg.get("runtime", 600)]
    reasons += verdict(m, pm, cfg.get("tol", 0.03))
    if cfg.get("note"):
        print("внимание:", cfg["note"])
    if not pm:
        print("ИТОГ: родитель не оценён — сравнить не с чем, смотри абсолютные числа")
    else:
        print("ИТОГ: ПРОШЛА БЫ GATE" if not reasons else "ИТОГ: НЕ ПРОШЛА — " + "; ".join(reasons))


if __name__ == "__main__":
    main()
