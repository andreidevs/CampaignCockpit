"""
Лаборатория версий стратегии. Агенту не нужен.

Версия = набор разрешённых настроек поверх agent.py (PATCH_TARGETS) + lineage.
Цикл: матрица тестов → детекторы проблем → патч (LLM через privacy gateway или шаблон)
→ новая draft-версия → та же матрица → gate против родителя → candidate/failed.
Promote — только по кнопке: константы переписываются в agent.py, submission.csv пересобирается.
AI-версия (ai_run: Claude Code / Codex правят копию agent.py в песочнице) несёт свой код в `source`;
её дети наследуют этот код, матрица грузит его в воркеры вместо agent.py.

    python3 lab.py      # self-check (во временной схеме Postgres, быстрая матрица)
"""

import ast
import asyncio
import difflib
import functools
import hashlib
import importlib
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

import agent
import db
import harness

ROOT = Path(__file__).parent
QUICK = os.environ.get("LAB_QUICK") == "1"
SEEDS = list(range(2 if QUICK else 10))
# главное семейство gate — больше миров: на 10 один мир около нуля решал исход «прогонов в минусе больше»
PRIMARY_SEEDS = list(range(2 if QUICK else 30))

# --- разрешённые зоны изменений (они же policy engine) ---------------------
PATCH_TARGETS = {
    "LCB_K": {"type": "float", "min": 0.2, "max": 1.5, "group": "scoring", "desc": "штраф за неуверенность: в план идёт mu − k·σ > 0"},
    "PRIOR_STD": {"type": "float", "min": 0.1, "max": 0.5, "group": "scoring", "desc": "ширина prior: насколько не доверяем истории"},
    "EI_STOP": {"type": "float", "min": 0.001, "max": 0.05, "group": "policy", "desc": "агрессивность разведки: стоп, когда EI < доли стартового"},
    "ARMS_PER_CELL": {"type": "int", "min": 2, "max": 6, "group": "experts", "desc": "гипотез prior на ячейку"},
    "LLM_PER_CELL": {"type": "int", "min": 1, "max": 3, "group": "experts", "desc": "гипотез LLM на ячейку"},
    "LLM_CLIP": {"type": "float", "min": 0.1, "max": 0.5, "group": "experts", "desc": "потолок |оценки| LLM"},
    "PILOT_FRAC": {"type": "float", "min": 0.02, "max": 0.2, "group": "pilots", "desc": "размер пилота как доля ячейки"},
    "PILOT_MIN": {"type": "int", "min": 10, "max": 200, "group": "pilots", "desc": "минимальный пилот"},
    "PILOT_MAX": {"type": "int", "min": 30, "max": 200, "group": "pilots", "desc": "максимальный пилот"},
    "PILOT_SIZING": {"type": "choice", "choices": ["fixed", "adaptive"], "group": "pilots", "desc": "adaptive: большой пилот только спорным рукавам"},
    "PILOT_BUDGET_SHARE": {"type": "float", "min": 0.02, "max": 1.0, "group": "policy", "desc": "максимальная доля бюджета на пилоты"},
    "PILOT_CONTACT_SHARE": {"type": "float", "min": 0.02, "max": 1.0, "group": "policy", "desc": "максимальная доля охвата на пилоты"},
    "PILOT_VALUE": {"type": "choice", "choices": ["ei", "voi"], "group": "policy", "desc": "voi: пилот, только если польза смены решения > цены пилота"},
    "LCB_K_PILOT": {"type": "float", "min": 0.0, "max": 1.5, "group": "scoring", "desc": "штраф за неуверенность у рукавов после пилота"},
    "LLM_GATE": {"type": "float", "min": 0.0, "max": 1.0, "group": "experts", "desc": "LLM-рукав только там, где лучший prior mu < порога"},
    "RANK_BY": {"type": "choice", "choices": ["mu", "lcb"], "group": "scoring", "desc": "ранжирование ячеек: по mu или по mu − k·σ"},
    "CHANNEL_MU": {"type": "choice", "choices": ["mu", "lcb"], "group": "channels", "desc": "ценность канала по mu или по консервативной оценке"},
    "UPGRADE_MIN_ROI": {"type": "float", "min": 0.0, "max": 5.0, "group": "channels", "desc": "no-call-under-threshold: апгрейд канала только при Δnet/Δcost ≥ порога"},
    "META_CONTROLLER": {"type": "bool", "group": "experts", "desc": "вес LLM = hit rate его пилотов"},
    "PRIOR_MODEL": {"type": "choice", "choices": ["hist", "catboost_shift", "catboost_full"], "group": "experts", "desc": "prior: только история, CatBoost-сдвиг или смесь модель/история"},
    "PRIOR_ALPHA_MAX": {"type": "float", "min": 0.0, "max": 1.0, "group": "experts", "desc": "catboost_full: вес модели при большой истории перехода"},
    "RISK_LAMBDA": {"type": "float", "min": 0.0, "max": 0.5, "group": "experts", "desc": "штраф prior за неуверенность: μ −= λ·(1/√(n+1) + доля OOD)"},
    "PRIOR_WEIGHT": {"type": "choice", "choices": ["rows", "arpu"], "group": "experts", "desc": "усреднение предсказаний модели по ячейке: по строкам или с весом ARPU"},
    "USE_LLM": {"type": "bool", "group": "policy", "desc": "llm-disabled mode"},
    "PRIVACY_MODE": {"type": "choice", "choices": ["aggregate_only", "synthetic_only", "debug_safe"], "group": "privacy", "desc": "что видит LLM"},
    "LLM_PROMPT_EXTRA": {"type": "str", "max_len": 500, "group": "prompt", "desc": "дополнение к инструкции LLM-эксперта"},
}
MUST = ("local_single", "schema", "guardrails", "runtime", "llm_disabled", "prompt_failure")
TH = {"cv": 0.5, "neg_pilots": 0.5, "sms_share": 0.8, "overlap": 0.1, "low_conf": 0.3, "runtime": 600, "gate_tol": 0.03}
# семейства миров для gate: тест → (ключ метрик, подпись). Главное — harsh0: по описанию среды стратегия
# без разведки там получает ~1/15 оракула, как в боевой среде; в stress_10 история почти верна (≈1/2 оракула)
WORLDS = {"harsh_0": ("harsh0", "жёсткие миры, история бесполезна"),
          "harsh_50": ("harsh50", "жёсткие миры, история наполовину верна"),
          "stress_10": ("stress", "стресс-миры")}
PRIMARY = "harsh0"
WORLDS_BY_KEY = {key: label for key, label in WORLDS.values()}


def validate(changes, base=None):
    """Отсекает всё вне PATCH_TARGETS и клипует значения. → (clean, errors)."""
    clean, errors = {}, []
    for k, v in (changes or {}).items():
        t = PATCH_TARGETS.get(k)
        if t is None:
            errors.append(f"{k}: не разрешённая зона изменений")
            continue
        try:
            if t["type"] in ("float", "int"):
                v = min(max(float(v), t["min"]), t["max"])
                v = int(round(v)) if t["type"] == "int" else round(v, 4)
            elif t["type"] == "bool":
                v = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes")
            elif t["type"] == "choice":
                if v not in t["choices"]:
                    raise ValueError(f"не из {t['choices']}")
            else:
                v = str(v)[: t["max_len"]].replace("\n", " ")
        except (TypeError, ValueError) as e:
            errors.append(f"{k}: {e}")
            continue
        clean[k] = v
    cfg = {**(base or {}), **clean}
    if cfg.get("PILOT_MIN", 0) > cfg.get("PILOT_MAX", 200):
        errors.append("PILOT_MIN > PILOT_MAX")
        clean.pop("PILOT_MIN", None)
    return clean, errors


# --- константы в тексте agent.py ------------------------------------------
def _find(text, key):
    """(match, value_src, tail) для строки `KEY = value  # коммент`; '#' внутри строки не ломает разбор."""
    m = re.search(rf"^{key} = (.*)$", text, re.M)
    if not m:
        raise KeyError(key)
    rest = m.group(1)
    for i in [j for j, c in enumerate(rest) if c == "#"] + [len(rest)]:
        src = rest[:i].rstrip()
        try:
            ast.literal_eval(src)
            return m, src, rest[len(src):]
        except (ValueError, SyntaxError):
            continue
    raise ValueError(f"{key}: не литерал")


def read_constants(text):
    return {k: ast.literal_eval(_find(text, k)[1]) for k in PATCH_TARGETS}


def set_constants(text, config):
    for k, v in config.items():
        m, _, tail = _find(text, k)
        lit = json.dumps(v, ensure_ascii=False) if isinstance(v, str) else repr(v)
        text = text[:m.start()] + f"{k} = {lit}{tail}" + text[m.end():]
    return text


def _sha(x):
    return hashlib.sha256(x.encode()).hexdigest()[:12]


def _commit():
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "agent.py"], cwd=ROOT, capture_output=True, text=True).stdout
        return h + ("-dirty" if dirty.strip() else "")
    except OSError:
        return None


# --- хранилище версий -------------------------------------------------------
_lock = threading.RLock()


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load(vid):
    with Session(db.engine) as s:
        v = s.get(db.LabVersion, vid)
    if v is None:
        raise FileNotFoundError(vid)  # server.py отвечает на это 404
    return v.data


def save(v):
    with _lock:
        db.upsert(db.LabVersion, id=v["id"], data=v)


def versions():
    with Session(db.engine) as s:
        return [r.data for r in s.scalars(select(db.LabVersion).order_by(db.LabVersion.id))]


def current_id():
    pro = [v for v in versions() if v.get("promoted")]
    return max(pro, key=lambda v: v.get("promoted_at") or "")["id"] if pro else None


def create(parent_id, changes, created_by="human", kind="manual", source=None, **meta):
    """source — новый код agent.py (AI-версия): константы берутся из него, иначе код и конфиг — от родителя."""
    with _lock:
        vs = versions()
        parent = load(parent_id) if parent_id else None
        src = source if source is not None else (parent or {}).get("source")
        base = (read_constants(source) if source is not None else parent["config"] if parent
                else read_constants((ROOT / "agent.py").read_text()))
        clean, errors = validate(changes, base)
        if parent and not clean and source is None:
            raise ValueError("; ".join(errors) or "нет изменений")
        cfg = {**base, **clean}
        v = {
            "id": f"v{len(vs) + 1:03d}", "parent_id": parent_id, "created_by": created_by, "kind": kind,
            "created_at": _now(), "commit_hash": _commit(),
            "prompt_hash": _sha(agent.LLM_INSTRUCTION + cfg["LLM_PROMPT_EXTRA"]),
            "config_hash": _sha(json.dumps(cfg, sort_keys=True)), "config": cfg,
            "diff": [{"key": k, "from": parent["config"].get(k), "to": cfg[k]} for k in cfg
                     if parent and parent["config"].get(k) != cfg[k]],
            "status": "draft", "promoted": False, "promoted_at": None, "rejected_changes": errors,
            "hypothesis": meta.get("hypothesis", ""), "rationale": meta.get("rationale", ""),
            "expected_benefit": meta.get("expected_benefit", ""), "issues_addressed": meta.get("issues_addressed", []),
            "template": meta.get("template"), "author": meta.get("author"), "ai": meta.get("ai"),
            "source": src, "source_sha": _sha(src) if src else None, "tests": [], "metrics": {}, "runs": [], "issues": [], "gate": None, "audit": [],
        }
        save(v)
        return v


def ensure_baseline():
    if not versions():
        v = create(None, {}, created_by="system", kind="baseline", hypothesis="Текущий agent.py",
                   rationale="Стартовая версия: константы, которые сейчас в agent.py")
        v.update(status="promoted", promoted=True, promoted_at=_now())
        save(v)


# --- один прогон (в процессе-воркере) ---------------------------------------
def _traced(base):
    class Traced(base):
        def _explore(self, env, cells, arms):
            self.cells, self.arms = cells, arms
            super()._explore(env, cells, arms)
    return Traced


_Traced = _traced(agent.Agent)


def _load_agent(source):
    """Модуль agent из текста версии. __file__ — как у agent.py: DATA_DIR и прочие пути остаются рабочими."""
    m = type(sys)("agent")
    m.__file__ = str(ROOT / "agent.py")
    exec(compile(source, f"agent@{_sha(source)}.py", "exec"), m.__dict__)
    return m


def _db_cached(call):
    """
    Один ответ LLM на промпт для всей лаборатории. Без этого gate сравнивал бы шум модели:
    тот же конфиг давал медиану стресс-миров 6.57M и 7.06M в двух прогонах.
    Версии с другим промптом (LLM_PROMPT_EXTRA, PRIVACY_MODE…) получают свой ответ.
    """
    def f(prompt, model=None):
        key = _sha(prompt if model is None else model + prompt)
        with Session(db.engine) as s:
            hit = s.get(db.LlmCache, key)
        if hit:
            return hit.response
        out = call(prompt, model)
        db.upsert(db.LlmCache, key=key, response=out)
        return out
    return f


def _init_worker(source=None):
    global agent, _Traced
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    db.engine.dispose(close=False)  # при fork соединения родителя не делим
    if source:  # код версии вместо agent.py: всё, что импортирует agent после этого, получает его
        m = _load_agent(source)
        m.DATA_DIR = agent.DATA_DIR  # при fork — подменённый сервером каталог загрузок
        sys.modules["agent"] = agent = m
        _Traced = _traced(m.Agent)
        if "stress_eval" in sys.modules:  # при fork уже импортирован со старым agent (его подклассы)
            importlib.reload(sys.modules["stress_eval"])
    agent._llm_call = functools.lru_cache(_db_cached(agent._llm_call))


def _apply(config):
    for k, v in config.items():
        setattr(agent, k, v)


def run_once(config, world, seed, test):
    import pandas as pd
    import stress_eval as se
    from environment import make_environment
    from mock_environment import CHANNELS, MAX_TOTAL_CONTACTS, TOTAL_BUDGET, _mock_fallback, _mock_impact_model
    from scoring_core import apply_filters, sanitize_campaigns, score_campaigns

    _apply(config)
    model = {"stress": lambda: se.world(seed), "harsh0": lambda: se.harsh_world(seed, 0.0),
             "harsh50": lambda: se.harsh_world(seed, 0.5)}.get(world, lambda: _mock_impact_model(se.history))()
    env, internals = make_environment(se.profile, model, se.dict_tariff, CHANNELS, TOTAL_BUDGET,
                                      MAX_TOTAL_CONTACTS, _mock_fallback, seed=seed)
    a, crash = _Traced(), None
    t = time.time()
    try:
        raw = a.act(env) or []
    except Exception as e:  # падение — тоже результат теста
        raw, crash = [], f"{type(e).__name__}: {e}"
    runtime = time.time() - t
    camps = sanitize_campaigns(raw, env.tariffs)[:10]
    pilots_raw = internals.executed_pilot_campaigns()
    df = pd.DataFrame(pilots_raw + camps)
    for c in se.COLS:
        if c not in df.columns:
            df[c] = None
    score = score_campaigns(df, se.profile, model, se.dict_tariff, se.profile["predicted_arpu"].sum(), _mock_fallback)
    names = {c["campaign_name"] for c in camps}
    detail = [d for d in score["campaigns_detail"] if d["name"] in names]
    by_name = {c["campaign_name"]: c for c in camps}

    ids = [set(apply_filters(se.profile, pd.Series({**{k: None for k in se.COLS}, **c}))["ID_NUMBER"]) for c in camps]
    total_ids = sum(len(s) for s in ids)
    arms, chosen = getattr(a, "arms", {}), getattr(a, "_chosen", [])
    planned = {(x["cur"], x["seg"], x["target"]) for x in chosen}
    experts = {}
    for e in ("prior", "llm"):
        mine = {k: m for k, m in arms.items() if e in m.get("src", ())}
        obs = [o for m in mine.values() for o in m.get("obs", [])]
        experts[e] = {"arms": len(mine), "piloted": sum(1 for m in mine.values() if m["n"]), "pilots": len(obs),
                      "hits": sum(o > 0 for o in obs), "planned": len(planned & mine.keys()),
                      "gain": sum(x["mu"] * CHANNELS[x["ch"]]["conversion_multiplier"] * x["S"]
                                  for x in chosen if e in arms.get((x["cur"], x["seg"], x["target"]), {}).get("src", ()))}
    return {
        "test": test, "world": world, "seed": seed, "llm": bool(config["USE_LLM"] and agent.llm_config()[1]),
        "net": float(score["net_arpu_gain"]), "raw_campaigns": len(raw), "n_campaigns": len(camps),
        "invalid": len(raw) - len(camps), "crash": crash, "runtime": round(runtime, 2),
        "fallback": any(l.startswith("fallback") for l in a.log) or any(c["campaign_name"].startswith("fallback") for c in camps),
        "pilots": len(env.pilot_history), "pilot_sizes": [int(r["n_customers"]) for r in env.pilot_history],
        "pilot_cost": float(sum(r["cost"] for r in env.pilot_history)),
        "pilot_obs": [float(o) for m in arms.values() for o in m.get("obs", [])],
        "piloted_arms": sum(1 for m in arms.values() if m["n"]), "piloted_planned": sum(1 for k in planned if arms.get(k, {}).get("n")),
        "total_contacts": int(score["total_contacts"]), "total_cost": float(score["total_cost"]),
        "caps": sorted({k for d in detail for k in ("capped_at_campaign_limit", "capped_at_reach_budget", "capped_at_money_budget") if d.get(k)}),
        "overlap": 1 - len(set().union(*ids)) / total_ids if total_ids else 0.0,
        "campaigns": [{"name": d["name"], "channel": d["channel"], "target": by_name[d["name"]]["target_tariff"],
                       "seg": by_name[d["name"]].get("filter_arpu_segment"), "n": int(d["n_contacts"]),
                       "gross": float(d["gross_lift"]), "cost": float(d["cost"])} for d in detail],
        "planned": [{"cur": x["cur"], "seg": x["seg"], "target": x["target"], "mu": float(x["mu"]), "sd": float(x.get("sd", 0)),
                     "S": float(x["S"]), "n": int(x["n"]), "ch": x["ch"]} for x in chosen],
        "experts": experts, "weights": dict(a.weights), "log": a.log, "audit": a.llm_audit,
    }


def run_check(config, name):
    """prompt_failure: мусор от LLM и падение API не ломают план; пустая история → всё равно ≥1 кампании."""
    import contextlib
    import io
    import stress_eval as se
    _apply({**config, "USE_LLM": True})
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            se.check_llm()
            se.check_never_empty()
        return {"check": name, "ok": True, "message": out.getvalue().strip()}
    except Exception as e:
        return {"check": name, "ok": False, "message": f"{type(e).__name__}: {e}\n{out.getvalue()}"}


def run_template(seed):
    import agent_template
    import stress_eval as se
    return float(se.run(agent_template.Agent, se.world(seed), seed))


# --- матрица тестов ---------------------------------------------------------
_pool = None
WORKERS = max(2, min(8, (os.cpu_count() or 2) - 1))


def pool():
    global _pool
    if _pool is None:
        _pool = ProcessPoolExecutor(max_workers=WORKERS, initializer=_init_worker)
    return _pool


def template_nets():
    with Session(db.engine) as ses:
        cache = {r.seed: r.data for r in ses.scalars(select(db.TemplateNet))}
    miss = [s for s in SEEDS if s not in cache]
    for s, r in zip(miss, pool().map(run_template, miss)):
        db.upsert(db.TemplateNet, seed=s, data=r)
        cache[s] = r
    return [cache[s] for s in SEEDS]


def _stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return {}
    return {"median": float(np.median(xs)), "min": float(min(xs)), "max": float(max(xs)),
            "positive": int(sum(x > 0 for x in xs)), "n": len(xs)}


def matrix(config, source=None):
    """source — код версии: свой пул на эту матрицу (модуль agent в воркере один на процесс)."""
    p = ProcessPoolExecutor(max_workers=WORKERS, initializer=_init_worker, initargs=(source,)) if source else pool()
    try:
        return _matrix(p, config)
    finally:
        if source:
            p.shutdown(cancel_futures=True)


def _matrix(p, config):
    jobs = ([("local_single", "mock", 42)] + [("local_10", "mock", s) for s in SEEDS]
            + [(t, WORLDS[t][0], s) for t in WORLDS for s in (PRIMARY_SEEDS if WORLDS[t][0] == PRIMARY else SEEDS)]
            + [("llm_disabled", "stress", s) for s in SEEDS[:3]])
    first = p.submit(run_once, config, *jobs[0][1:], jobs[0][0]).result()  # наполняет кэш LLM до параллельных прогонов
    jobs = jobs[1:]
    futs = [p.submit(run_once, {**config, "USE_LLM": False} if t == "llm_disabled" else config, w, s, t) for t, w, s in jobs]
    chk = p.submit(run_check, config, "prompt_failure")
    runs = [first] + [f.result() for f in futs]
    check = chk.result()
    tmpl = template_nets()

    by = lambda t: [r for r in runs if r["test"] == t]  # noqa: E731
    nets = lambda t: [r["net"] for r in by(t)]  # noqa: E731
    valid = lambda r: r["invalid"] == 0 and 1 <= r["n_campaigns"] <= 10 and not r["crash"]  # noqa: E731
    guard = lambda r: (r["pilots"] <= 20 and all(10 <= n <= 200 for n in r["pilot_sizes"])  # noqa: E731
                       and r["total_contacts"] <= 15000 and r["total_cost"] <= 100000 + 1e-6 and not r["caps"])
    st = _stats(nets("stress_10"))
    rt = [r["runtime"] for r in runs]
    tests = [
        {"name": "local_single", "title": "local_eval: один прогон (seed 42)", "passed": all(r["net"] > 0 and r["pilots"] > 0 and valid(r) for r in by("local_single")),
         "stats": _stats(nets("local_single")), "detail": "net > 0, пилоты > 0, план валиден"},
        {"name": "local_10", "title": f"local_eval --runs {len(SEEDS)}", "passed": all(n > 0 for n in nets("local_10")),
         "stats": _stats(nets("local_10")), "detail": "все прогоны в плюс"},
        {"name": "stress_10", "title": f"stress_eval --runs {len(SEEDS)}", "passed": all(n > 0 for n in nets("stress_10")) and st.get("median", 0) > np.median(tmpl),
         "stats": st, "detail": f"все миры в плюс и медиана лучше шаблона ({np.median(tmpl) / 1e6:.2f}M)"},
        *[{"name": t, "title": f"{label} ({len(nets(t))})", "passed": all(n > 0 for n in nets(t)), "stats": _stats(nets(t)),
           "detail": "все миры в плюс; в gate медиана " + ("не должна упасть" if key == PRIMARY else f"не должна упасть больше чем на {TH['gate_tol']:.0%}")}
          for t, (key, label) in WORLDS.items() if t != "stress_10"],
        {"name": "schema", "title": "Валидация кампаний", "passed": all(valid(r) for r in runs),
         "stats": {"invalid": sum(r["invalid"] for r in runs), "crashes": sum(bool(r["crash"]) for r in runs)}, "detail": "1–10 кампаний, 0 отброшенных, без падений"},
        {"name": "guardrails", "title": "Бюджет и охват", "passed": all(guard(r) for r in runs),
         "stats": {"max_contacts": max(r["total_contacts"] for r in runs), "max_cost": max(r["total_cost"] for r in runs),
                   "max_pilots": max(r["pilots"] for r in runs)}, "detail": "≤20 пилотов по 10–200, ≤15000 контактов, ≤100000 у.е., без обрезки"},
        {"name": "runtime", "title": "Время работы", "passed": max(rt) < TH["runtime"],
         "stats": {"median": float(np.median(rt)), "max": max(rt)}, "detail": f"каждый прогон < {TH['runtime']} с"},
        {"name": "llm_disabled", "title": "LLM выключен", "passed": all(valid(r) and r["net"] > 0 for r in by("llm_disabled")),
         "stats": _stats(nets("llm_disabled")), "detail": "USE_LLM=False: план валиден и в плюс"},
        {"name": "prompt_failure", "title": "Сбой и мусор LLM", "passed": check["ok"], "stats": {}, "detail": check["message"]},
    ]
    for t in tests:
        t["must"], t["passed"] = t["name"] in MUST, bool(t["passed"])  # numpy bool → json
    return tests, runs


MAIN = ("local_10", *WORLDS)


def metrics(runs):
    main = [r for r in runs if r["test"] in MAIN]
    local = [r["net"] for r in runs if r["test"] == "local_10"]
    obs = [o for r in main for o in r["pilot_obs"]]
    gain = {e: sum(r["experts"][e]["gain"] for r in main) for e in ("prior", "llm")}
    tot = sum(gain.values()) or 1
    return {
        **{key: _stats([r["net"] for r in runs if r["test"] == t]) for t, (key, _) in WORLDS.items()}, "local": _stats(local),
        "negative_runs": sum(r["net"] <= 0 for r in main), "invalid": sum(r["invalid"] for r in runs),
        "pilots_mean": float(np.mean([r["pilots"] for r in main])), "pilot_cost_mean": float(np.mean([r["pilot_cost"] for r in main])),
        "pilot_hit_rate": float(np.mean([o > 0 for o in obs])) if obs else None,
        "pilot_to_plan": float(sum(r["piloted_planned"] for r in main) / max(1, sum(r["piloted_arms"] for r in main))),
        "diversity": float(np.mean([len({(c["target"], c["channel"]) for c in r["campaigns"]}) for r in main])),
        "overlap": float(np.mean([r["overlap"] for r in main])),
        "runtime_max": max(r["runtime"] for r in runs),
        "expert_share": {e: g / tot for e, g in gain.items()},
        "llm_used": any(r["llm"] for r in runs),
    }


# --- детекторы проблем ------------------------------------------------------
def _m(x):
    return f"{x / 1e6:+.2f}M"


def detect(runs):
    main = [r for r in runs if r["test"] in MAIN]
    out = []

    def add(code, severity, title, evidence, metric=None):
        out.append({"code": code, "severity": severity, "title": title, "evidence": evidence, "metric": metric,
                    "templates": [t for t, m in TEMPLATES.items() if code in m["issues"]]})

    parts, any_neg, worst = [], False, 0.0
    for t, (_, label) in WORLDS.items():
        fam = [r for r in runs if r["test"] == t]
        if not fam:
            continue
        nets = [r["net"] for r in fam]
        med, neg = float(np.median(nets)), [r["seed"] for r in fam if r["net"] <= 0]
        cv = float(np.std(nets) / abs(med)) if med else float("inf")
        if neg:
            parts.append(f"{label}: медиана {_m(med)}, но {len(neg)}/{len(nets)} в минусе (seeds {neg}), минимум {_m(min(nets))}")
        elif cv > TH["cv"]:
            parts.append(f"{label}: σ/медиана = {cv:.2f} > {TH['cv']}, от {_m(min(nets))} до {_m(max(nets))}")
        any_neg, worst = any_neg or bool(neg), max(worst, cv)
    if parts:
        add("HIGH_VARIANCE_BETWEEN_SEEDS", "high" if any_neg else "medium", "Большой разброс между мирами", "; ".join(parts), worst)
    obs = [o for r in main for o in r["pilot_obs"]]
    if obs and np.mean([o < 0 for o in obs]) > TH["neg_pilots"]:
        k = sum(o < 0 for o in obs)
        add("TOO_MANY_NEGATIVE_PILOTS", "medium", "Слишком много пилотов в минус",
            f"{k}/{len(obs)} пилотов ({100 * k / len(obs):.0f}%) показали отрицательный uplift — бюджет разведки уходит на проигравшие гипотезы",
            k / len(obs))
    llm = [r["experts"]["llm"] for r in main]
    if llm and sum(e["arms"] for e in llm):
        dead = sum(e["planned"] == 0 for e in llm)
        if dead >= len(llm) / 2:
            add("WEAK_LLM_CONTRIBUTION", "medium", "LLM почти не влияет на план",
                f"LLM добавил в среднем {np.mean([e['arms'] for e in llm]):.0f} рукавов, пилотов на них {sum(e['pilots'] for e in llm)}, "
                f"но в {dead}/{len(llm)} прогонах ни один не попал в финальный план", dead / len(llm))
    contacts = [(sum(c["n"] for c in r["campaigns"] if c["channel"] == "sms"), sum(c["n"] for c in r["campaigns"])) for r in main]
    share = sum(a for a, _ in contacts) / max(1, sum(b for _, b in contacts))
    if share > TH["sms_share"]:
        add("OVERUSE_SMS", "low", "План почти целиком на SMS", f"{100 * share:.0f}% контактов плана — SMS, остальные каналы не используются", share)
    calls = [(r["seed"], c) for r in main for c in r["campaigns"] if c["channel"] == "call"]
    bad = [(s, c) for s, c in calls if c["gross"] < c["cost"]]
    val = [x["S"] / x["n"] for r in main for x in r["planned"]]
    low = [x for r in main for x in r["planned"] if x["ch"] == "call" and val and x["S"] / x["n"] < np.median(val)]
    if bad or low:
        add("CALL_CHANNEL_NOT_COST_EFFECTIVE", "high" if bad else "medium", "Звонок не окупается",
            (f"{len(bad)}/{len(calls)} кампаний через звонок: прирост меньше стоимости, напр. {bad[0][1]['name']} (seed {bad[0][0]}): "
             f"{_m(bad[0][1]['gross'])} при затратах {_m(bad[0][1]['cost'])}" if bad else "")
            + ("; " if bad and low else "") + (f"звонок назначен {len(low)} ячейкам с ценностью ниже медианы" if low else ""),
            len(bad) + len(low))
    ov = float(np.mean([r["overlap"] for r in main])) if main else 0
    if ov > TH["overlap"]:
        add("CAMPAIGN_OVERLAP_TOO_HIGH", "medium", "Кампании пересекаются", f"{100 * ov:.0f}% контактов плана — повторы одних и тех же абонентов", ov)
    pl = [x for r in main for x in r["planned"]]
    S = sum(x["S"] for x in pl) or 1
    weak = sum(x["S"] for x in pl if x["sd"] > abs(x["mu"])) / S
    if weak > TH["low_conf"]:
        add("LOW_POSTERIOR_CONFIDENCE", "medium", "План держится на неуверенных оценках",
            f"{100 * weak:.0f}% ARPU плана — ячейки, где σ больше |μ|: эффект может оказаться нулевым", weak)
    fb = [f"{r['test']}#{r['seed']}" for r in runs if r["fallback"] or r["crash"]]
    if fb:
        add("FALLBACK_TRIGGERED", "high", "Сработал fallback", f"{len(fb)} прогонов ушли в запасной план: {', '.join(fb[:5])}", len(fb))
    if main:
        tg = np.median([len({c["target"] for c in r["campaigns"]}) for r in main])
        chn = np.median([len({c["channel"] for c in r["campaigns"]}) for r in main])
        if tg <= 1 or chn <= 1:
            add("NO_DIVERSITY_IN_CAMPAIGNS", "low", "Кампании однотипны", f"медиана: {tg:.0f} целевых тарифов, {chn:.0f} каналов на план", tg)
    order = {"high": 0, "medium": 1, "low": 2}
    return sorted(out, key=lambda i: order[i["severity"]])


# --- шаблоны патчей ---------------------------------------------------------
TEMPLATES = {
    "conservative": {"title": "Усилить консервативный штраф", "issues": ["HIGH_VARIANCE_BETWEEN_SEEDS", "LOW_POSTERIOR_CONFIDENCE"],
                     "changes": lambda c: {"LCB_K": c["LCB_K"] + 0.25}},
    "adaptive_pilots": {"title": "Уменьшить ранние пилоты: адаптивный размер", "issues": ["TOO_MANY_NEGATIVE_PILOTS", "HIGH_VARIANCE_BETWEEN_SEEDS"],
                        "changes": lambda c: {"PILOT_SIZING": "adaptive"}},
    "risk_rank": {"title": "Ранжировать по risk-adjusted оценке", "issues": ["LOW_POSTERIOR_CONFIDENCE", "HIGH_VARIANCE_BETWEEN_SEEDS"],
                  "changes": lambda c: {"RANK_BY": "lcb"}},
    "clip_channels": {"title": "Обрезать рискованные апгрейды канала", "issues": ["CALL_CHANNEL_NOT_COST_EFFECTIVE", "OVERUSE_SMS"],
                      "changes": lambda c: {"CHANNEL_MU": "lcb", "UPGRADE_MIN_ROI": max(1.0, c["UPGRADE_MIN_ROI"])}},
    "llm_validation": {"title": "Строже валидировать ответ LLM", "issues": ["WEAK_LLM_CONTRIBUTION"],
                       "changes": lambda c: {"LLM_CLIP": c["LLM_CLIP"] - 0.2, "LLM_PER_CELL": 1}},
    "meta": {"title": "Мета-контроллер весов экспертов", "issues": ["WEAK_LLM_CONTRIBUTION", "TOO_MANY_NEGATIVE_PILOTS"],
             "changes": lambda c: {"META_CONTROLLER": True}},
    "more_arms": {"title": "Больше гипотез на ячейку", "issues": ["NO_DIVERSITY_IN_CAMPAIGNS"],
                  "changes": lambda c: {"ARMS_PER_CELL": c["ARMS_PER_CELL"] + 1}},
    "llm_off": {"title": "Выключить LLM-эксперта", "issues": ["FALLBACK_TRIGGERED", "WEAK_LLM_CONTRIBUTION"],
                "changes": lambda c: {"USE_LLM": False}},
}


def _candidates(parent):
    """Шаблоны по убыванию важности issues, затем остальные; без no-op и без уже испробованных конфигов."""
    seen = {v["config_hash"] for v in versions()}
    order = [t for i in parent.get("issues", []) for t in i["templates"]] + list(TEMPLATES)
    out = []
    for t in dict.fromkeys(order):
        ch, _ = validate(TEMPLATES[t]["changes"](parent["config"]), parent["config"])
        ch = {k: v for k, v in ch.items() if parent["config"].get(k) != v}
        cfg = {**parent["config"], **ch}
        if ch and _sha(json.dumps(cfg, sort_keys=True)) not in seen:
            out.append((t, ch))
    return out


REMEDIATION_INSTRUCTION = (
    "Ты инженер, который улучшает агента маркетинговых кампаний. В контексте: issues — найденные проблемы версии "
    "с доказательствами, metrics — сводные метрики тестов, targets — разрешённые настройки (текущее значение, границы), "
    "templates — готовые безопасные патчи. Выбери ОДНО небольшое изменение (1–2 настройки), которое скорее всего "
    "исправит самую важную проблему. Главный критерий приёмки — медиана harsh0 (миры, где история бесполезна, "
    "как в боевой среде) не должна упасть, худший мир harsh0 (harsh0_min) не должен стать хуже; stress и harsh50 — "
    "не больше чем на 3%. Менять можно только ключи из targets. "
    'Ответ строго JSON: {"template": "id или null", "changes": {"KEY": value}, "hypothesis": "...", "expected_benefit": "..."}'
)


def propose(parent):
    """Готовит патч и создаёт draft-версию. None — пробовать больше нечего."""
    cands = _candidates(parent)
    if not cands:
        return None
    issues = parent.get("issues", [])
    t, ch = cands[0]
    meta = {"template": t, "created_by": "template", "hypothesis": TEMPLATES[t]["title"],
            "rationale": "Шаблон для: " + (", ".join(i["code"] for i in issues if t in i["templates"]) or "без issues — проверка улучшения"),
            "expected_benefit": "устранить: " + ", ".join(i["code"] for i in issues if t in i["templates"])}
    audit = []
    if agent.llm_config()[1]:
        m = parent["metrics"]
        ctx = {
            "issues": [{"code": i["code"], "severity": i["severity"], "evidence": i["evidence"]} for i in issues],
            "metrics": {"harsh0_median": m.get("harsh0", {}).get("median"), "harsh0_min": m.get("harsh0", {}).get("min"), "harsh50_median": m.get("harsh50", {}).get("median"),
                        "stress_median": m["stress"].get("median"), "stress_min": m["stress"].get("min"),
                        "negative_runs": m["negative_runs"], "pilot_hit_rate": m["pilot_hit_rate"], "diversity": m["diversity"]},
            "targets": {k: {"value": parent["config"][k], "min": t_.get("min"), "max": t_.get("max"), "choices": t_.get("choices"),
                            "desc": t_["desc"]} for k, t_ in PATCH_TARGETS.items() if k != "LLM_PROMPT_EXTRA"},
            "templates": [{"id": tid, "title": TEMPLATES[tid]["title"], "changes": c} for tid, c in cands[:6]],
        }
        allowed = ({"issues", "code", "severity", "evidence", "metrics", "targets", "value", "min", "max", "choices", "desc",
                    "templates", "id", "title", "changes"} | set(ctx["metrics"]) | set(PATCH_TARGETS))
        try:
            ans = json.loads(agent.llm_proxy("remediation", ctx, REMEDIATION_INSTRUCTION, audit, allowed))
            llm_ch, errors = validate(ans.get("changes") or {}, parent["config"])
            llm_ch = {k: v for k, v in llm_ch.items() if parent["config"].get(k) != v}
            cfg_hash = _sha(json.dumps({**parent["config"], **llm_ch}, sort_keys=True))
            if llm_ch and cfg_hash not in {v["config_hash"] for v in versions()}:
                ch = llm_ch
                meta.update(created_by="llm", template=ans.get("template") if ans.get("template") in TEMPLATES else None,
                             hypothesis=str(ans.get("hypothesis", ""))[:300], expected_benefit=str(ans.get("expected_benefit", ""))[:300],
                             rationale="LLM выбрал патч по issues" + (f"; отброшено: {errors}" if errors else ""))
            audit[-1]["parsed"] = [{"changes": llm_ch, "accepted": meta["created_by"] == "llm"}]
            audit[-1]["rejected"] = [{"row": e, "reason": "validate"} for e in errors]
        except Exception as e:  # LLM недоступен или ответил мусором — остаётся шаблон
            meta["rationale"] += f" (LLM: {type(e).__name__})"
        with Session(db.engine) as s, s.begin():
            s.add_all(db.LabAudit(data={**r, "parent_id": parent["id"], "at": _now()}) for r in audit)
    v = create(parent["id"], ch, kind="auto", issues_addressed=[i["code"] for i in issues if not meta["template"] or meta["template"] in i["templates"]],
               **meta)
    return v


# --- gate / evaluate / remediate / promote ----------------------------------
def gate(v, parent):
    reasons = [f"must-have не пройден: {t['name']}" for t in v["tests"] if t["must"] and not t["passed"]]
    m = v["metrics"]
    if m["invalid"]:
        reasons.append(f"отброшенных кампаний: {m['invalid']}")
    if parent and parent.get("metrics"):
        pm = parent["metrics"]
        for key, label in WORLDS.values():
            if key not in pm or key not in m:
                reasons.append(f"нет результатов «{label}» у {'родителя' if key not in pm else 'версии'} — перепрогони матрицу")
                continue
            a, b = pm[key]["median"], m[key]["median"]
            tol = 0.0 if key == PRIMARY else TH["gate_tol"]  # главное семейство — без допуска, остальные ±шум
            if b < a - tol * abs(a):
                reasons.append(f"медиана «{label}» упала: {_m(a)} → {_m(b)}" + (f" (допуск {tol:.0%})" if tol else ""))
        # худший мир, а не число миров в минусе: счётчик на границе нуля — монетка
        # (адаптивные пилоты: медиана +70%, худший мир −2.85M → −0.99M, но «в минусе» 1 → 2)
        if PRIMARY in pm and PRIMARY in m and m[PRIMARY]["min"] < pm[PRIMARY]["min"]:
            reasons.append(f"худший мир «{WORLDS_BY_KEY[PRIMARY]}» стал хуже: {_m(pm[PRIMARY]['min'])} → {_m(m[PRIMARY]['min'])}")
    return {"passed": not reasons, "reasons": reasons, "vs": parent["id"] if parent and parent.get("metrics") else None}


def evaluate(vid):
    v = load(vid)
    v["status"] = "evaluating"
    save(v)
    try:
        tests, runs = matrix(v["config"], v.get("source"))
        v.pop("error", None)
        v.update(tests=tests, runs=runs, metrics=metrics(runs), issues=detect(runs), evaluated_at=_now(),
                 audit=list({r["prompt_sha"]: {k: x for k, x in r.items()} for run in runs for r in run["audit"]}.values()))
        for r in v["runs"]:
            r.pop("audit")
        parent = load(v["parent_id"]) if v["parent_id"] else None
        if parent and any(key not in parent.get("metrics", {}) for key, _ in WORLDS.values()):
            parent = evaluate(parent["id"])  # родитель оценён старой матрицей — сравниваем на равных
        v["gate"] = gate(v, parent)
        v["status"] = "promoted" if v["promoted"] else "candidate" if v["gate"]["passed"] else "failed"
    except Exception as e:
        v.update(status="promoted" if v["promoted"] else "failed", error=f"{type(e).__name__}: {e}\n{traceback.format_exc()[-2000:]}")
    save(v)
    return v


def remediate(vid, steps=1):
    parent = load(vid)
    if not parent.get("metrics"):
        parent = evaluate(vid)
    made = []
    for _ in range(steps):
        child = propose(parent)
        if child is None:
            break
        child = evaluate(child["id"])
        made.append(child["id"])
        if child["gate"] and child["gate"]["passed"]:
            parent = child
    return made


def promote(vid):
    v = load(vid)
    if v["status"] != "candidate":
        raise ValueError(f"promote только из candidate, сейчас {v['status']}")
    if getattr(sys, "frozen", False):
        raise ValueError("promote недоступен в десктоп-версии: agent.py внутри приложения только для чтения")
    path = ROOT / "agent.py"
    text = set_constants(v.get("source") or path.read_text(), v["config"])
    assert read_constants(text) == v["config"], "константы не записались"
    path.write_text(text)
    env = {k: x for k, x in os.environ.items() if k not in ("OPENAI_API_KEY", "OPENROUTER_API_KEY")}  # submission.csv воспроизводим без ключа
    r = subprocess.run([sys.executable, "make_submission.py"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    v.update(status="promoted", promoted=True, promoted_at=_now(), submission=r.stdout[-2000:] + r.stderr[-2000:], commit_hash=_commit(),
             needs_restart=bool(v.get("source")))  # код в этом процессе не перезагружаем — только константы
    save(v)
    return v


# --- фоновая очередь --------------------------------------------------------
_q = queue.Queue()
_worker = None


def enqueue(fn, *args):
    global _worker
    if _worker is None:
        def loop():
            while True:
                f, a = _q.get()
                try:
                    f(*a)
                except Exception:
                    traceback.print_exc()
                finally:
                    _q.task_done()
        _worker = threading.Thread(target=loop, daemon=True)
        _worker.start()
    _q.put((fn, args))


def mark_evaluating(vid):
    v = load(vid)
    v["status"] = "evaluating"
    save(v)


# --- код версии и AI-агент (Claude Code / Codex) ----------------------------
def effective_source(v):
    """agent.py, который реально исполняет версия: её код (или репозиторный) с её константами."""
    return set_constants(v.get("source") or (ROOT / "agent.py").read_text(), v["config"])


def code_diff(vid):
    v = load(vid)
    p = load(v["parent_id"]) if v["parent_id"] else None
    a = effective_source(p) if p else (ROOT / "agent.py").read_text()
    return "".join(difflib.unified_diff(a.splitlines(True), effective_source(v).splitlines(True),
                                        f"{p['id'] if p else 'repo'}/agent.py", f"{vid}/agent.py"))


AI_TIMEOUT = 1800
AI_LOG_MAX = 60_000
# что видит AI-агент в песочнице: организаторский пакет для локальных прогонов + документация
SANDBOX_FILES = ("environment.py", "scoring_core.py", "mock_environment.py", "local_eval.py", "agent_template.py", "stress_eval.py",
                 "lab_check.py")
SANDBOX_DOCS = ("agent.md", "experiments.md", "lab.md")
AI_RULES = """
## Правила
- Меняй только `agent.py` в текущем каталоге. Остальные файлы — организаторский пакет и справка, их правки выбрасываются.
- Интерфейс не ломай: `Agent().act(env)` возвращает список кампаний; константы верхнего уровня (`LCB_K = ...` и т.д.)
  не удаляй и не переименовывай — лаборатория читает их из кода.
- Проверка (зависимости уже стоят; разрешены только эти скрипты и `tail`/`head`/`grep`/`ls`/`cat`/`wc`):
  - `python3 lab_check.py` — **локальный gate**: те же миры и seed, что у gate лаборатории, сравнение с родителем,
    последняя строка — вердикт, ~10 с. **Обязателен перед тем, как закончить.**
  - для быстрых проб по ходу: `python3 stress_eval.py --keep 0 --runs 5 | tail -30` (harsh0), `python3 local_eval.py | tail -20`.
- Заканчивай, только когда `lab_check.py` пишет «ПРОШЛА БЫ GATE». Не получилось за разумное число попыток — верни
  agent.py к исходнику (`agent_parent.py`) и опиши, что пробовал и почему не вышло: отклонённая правка стоит полный
  прогон матрицы, честное «не вышло» — ничего.
- Не повторяй идеи из журнала ниже, отклонённые gate; если берёшь похожую — объясни, чем твоя отличается.
- Сетевые LLM-ключи недоступны: агент работает в режиме без LLM, это нормально.
- В конце кратко напиши: что изменил, почему это должно улучшить метрики, какой риск.
"""
_ai_lock = threading.Lock()  # один запуск агента за раз
_live = {}  # run_id → запуск, пока идёт: события пишутся сюда, в БД — не чаще раза в 2 с и в конце
_flushed = {}
_cancel = {}  # run_id → (loop, task) для остановки из другого потока
RUN_EVENTS_MAX = 3000
DIFF_FEEDBACK_MAX = 6000


def ai_busy():
    return _ai_lock.locked()


def run_save(r):
    db.upsert(db.AiRun, id=r["id"], data=r)
    _flushed[r["id"]] = time.time()


def run_load(rid):
    if rid in _live:
        return _live[rid]
    with Session(db.engine) as s:
        row = s.get(db.AiRun, rid)
    if row is None:
        raise FileNotFoundError(rid)
    return row.data


def runs():
    with Session(db.engine) as s:
        rows = [r.data for r in s.scalars(select(db.AiRun).order_by(db.AiRun.id.desc()))]
    return [_live.get(r["id"], r) for r in rows]


def ai_recover():
    """После перезапуска сервера «идущие» запуски мертвы: их поток и CLI умерли вместе с процессом."""
    for r in runs():
        if r["status"] == "running":
            if r.get("pid") and harness.alive_cli(r["pid"]):  # CLI пережил сервер — добиваем вместе с детьми
                harness.kill_tree(r["pid"])
            if r.get("box"):
                shutil.rmtree(r["box"], ignore_errors=True)
            r.update(status="failed", error="сервер перезапущен во время запуска", finished_at=_now())
            run_save(r)


def ai_start(parent_id, harness_id, model, task, author, steps=1, budget_usd=None):
    """
    Запуск в фоновом потоке, не привязан к HTTP-соединению. steps > 1 — автономный цикл:
    правка → матрица → gate; принята — следующий шаг от неё, отклонена — агенту причины и diff неудачной попытки.
    RuntimeError — уже идёт другой запуск.
    """
    if not _ai_lock.acquire(blocking=False):
        raise RuntimeError("AI-агент уже работает")
    try:
        load(parent_id)
        with Session(db.engine) as s:
            n = s.scalar(select(func.count()).select_from(db.AiRun))
        r = {"id": f"r{n + 1:03d}", "status": "running", "parent_id": parent_id, "harness": harness_id, "model": model or None,
             "task": task, "author": author, "steps": steps, "budget_usd": budget_usd, "step": 0, "created_at": _now(),
             "finished_at": None, "cost_usd": 0.0, "tokens_in": 0, "tokens_out": 0, "versions": [], "accepted": [],
             "best": parent_id, "error": None, "note": None, "summary": None, "n_events": 0, "events": []}
        run_save(r)
        _live[r["id"]] = r
        threading.Thread(target=_run_thread, args=(r,), daemon=True).start()
        return r
    except BaseException:
        _ai_lock.release()
        raise


def ai_cancel(rid):
    loop, task = _cancel.get(rid, (None, None))
    if loop is None:
        return False
    loop.call_soon_threadsafe(task.cancel)
    return True


def _run_thread(r):
    try:
        asyncio.run(_run(r))
    except Exception as e:  # не должно случаться: _run ловит своё; но замок и статус — в любом случае
        r.update(status="failed", error=f"{type(e).__name__}: {e}")
    finally:
        r["finished_at"] = r["finished_at"] or _now()
        run_save(r)
        _live.pop(r["id"], None)
        _cancel.pop(r["id"], None)
        _ai_lock.release()


def _emit(r, ev):
    r["n_events"] += 1
    r["events"].append({**ev, "i": r["n_events"], "step": r["step"]})
    del r["events"][:-RUN_EVENTS_MAX]
    if time.time() - _flushed.get(r["id"], 0) > 2:
        run_save(r)


class _StepFailed(Exception):
    """CLI не отработал (нет входа, квота, таймаут) — продолжать цикл бессмысленно."""


class _BudgetOut(Exception):
    """Claude остановился по --max-budget-usd посреди шага: правку не берём, цикл закончен штатно."""


def _brief(v):
    m = v.get("metrics") or {}
    return {k: {x: (m.get(k) or {}).get(x) for x in ("median", "min")} for k in ("harsh0", "harsh50", "stress")}


def _mln(v):
    return json.dumps({k: {x: None if y is None else round(y / 1e6, 3) for x, y in d.items()} for k, d in _brief(v).items()})


def _step_task(goal, parent, prev, step):
    """Задача шага: цель + что случилось на прошлом шаге (принято / отклонено gate / не собралось)."""
    if step == 1:
        return goal
    out = [goal, "", f"Это шаг {step} автономного цикла улучшения. Текущая база — {parent['id']}."]
    if isinstance(prev, str):
        out.append(f"Прошлая попытка не прошла проверку кода: {prev}. Исправь подход.")
    elif prev and prev["id"] != parent["id"]:
        g = prev.get("gate") or {}
        diff = code_diff(prev["id"])
        out += [f"Прошлая попытка {prev['id']} отклонена: " + ("; ".join(g.get("reasons", [])) or prev.get("error") or prev["status"]) + ".",
                f"Метрики попытки, млн: {_mln(prev)}; базы: {_mln(parent)}.",
                "Её правка (не повторяй — попробуй другую идею или исправь причину провала):",
                "```diff\n" + diff[:DIFF_FEEDBACK_MAX] + ("\n… (обрезано)" if len(diff) > DIFF_FEEDBACK_MAX else "") + "\n```"]
    else:
        out.append(f"Прошлый шаг принят: {parent['id']} прошла gate. Улучшай дальше от неё — следующей идеей, а не повтором.")
    return "\n".join(out)


async def _step(r, parent, task):
    """Один проход CLI. ("version", vid) | ("nochange", None) | ("invalid", причина); _StepFailed — CLI не отработал."""
    box = _sandbox(parent, task)
    try:
        before = (box / "agent.py").read_text()
        _emit(r, {"kind": "started", "parent": parent["id"], "harness": r["harness"], "model": r["model"] or ""})
        prompt = (f"Задача: {task}\n\nКонтекст (метрики родителя, правила) — в LAB_TASK.md в текущем каталоге. "
                  "Делай ровно то, что просит задача; если она не просит менять код — не меняй.")
        log, answer, failed, cost, tin, tout, budget_out = [], "", None, None, 0, 0, False
        left = r["budget_usd"] - r["cost_usd"] if r["budget_usd"] else None
        def started(pid):
            r.update(pid=pid, box=str(box))
            run_save(r)
        async for ev in harness.run(r["harness"], r["model"] or "", prompt, box, AI_TIMEOUT, started, left):
            if ev.get("detail"):  # пути песочницы → относительные (macOS отдаёт и /var, и /private/var)
                ev = {**ev, "detail": ev["detail"].replace(f"{box.resolve()}/", "").replace(f"{box}/", "")}
            if ev["kind"] == "text":
                answer = ev["text"]
            elif ev["kind"] == "result":
                answer = ev["text"] or answer
                failed = None if ev["ok"] else ev["text"] or "агент завершился с ошибкой"
                budget_out = bool(ev.get("budget"))
                cost, tin, tout = ev.get("cost_usd"), ev.get("tokens_in", 0), ev.get("tokens_out", 0)
                r["cost_usd"] += cost or 0
                r["tokens_in"] += tin
                r["tokens_out"] += tout
            elif ev["kind"] == "error":
                failed = ev["message"]
            _emit(r, ev)
            log.append(ev.get("text") or f"— {ev.get('tool', ev['kind'])} {ev.get('detail') or ev.get('message') or ''}".rstrip())
        r["summary"] = answer or r["summary"]
        if budget_out:
            raise _BudgetOut()
        if failed:
            raise _StepFailed(failed)
        after = (box / "agent.py").read_text()
    finally:
        shutil.rmtree(box, ignore_errors=True)
        r.update(pid=None, box=None)
    if after == before:
        return "nochange", answer
    try:
        compile(after, "agent.py", "exec")
        read_constants(after)
    except SyntaxError as e:
        return "invalid", f"SyntaxError: {e}"
    except (KeyError, ValueError) as e:
        return "invalid", f"в agent.py пропала или сломалась константа {e}"
    v = create(parent["id"], {}, created_by="ai", kind="ai", source=after, author=r["author"],
               hypothesis=r["task"][:300] if r["steps"] == 1 else f"{r['task'][:250]} · шаг {r['step']}", rationale=answer[:2000],
               ai={"harness": r["harness"], "model": r["model"], "task": task, "run_id": r["id"], "step": r["step"],
                   "cost_usd": cost, "tokens_in": tin, "tokens_out": tout, "log": "\n".join(log)[-AI_LOG_MAX:]})
    return "version", v["id"]


async def _run(r):
    _cancel[r["id"]] = (asyncio.get_running_loop(), asyncio.current_task())
    parent, prev = load(r["parent_id"]), None
    try:
        if not parent.get("metrics"):  # без метрик родителя нет ни lab_check, ни gate
            _emit(r, {"kind": "info", "text": f"{parent['id']} ещё не оценена — прогоняю матрицу"})
            parent = await asyncio.to_thread(evaluate, parent["id"])
        for step in range(1, r["steps"] + 1):
            if r["budget_usd"] and r["cost_usd"] >= r["budget_usd"]:
                r["note"] = f"бюджет ${r['budget_usd']:g} исчерпан (${r['cost_usd']:.2f})"
                _emit(r, {"kind": "info", "text": r["note"]})
                break
            r["step"] = step
            if r["steps"] > 1:
                _emit(r, {"kind": "step", "text": f"Шаг {step} из {r['steps']} · база {parent['id']}"})
            outcome, x = await _step(r, parent, _step_task(r["task"], parent, prev, step))
            if outcome == "nochange":
                r["note"] = "агент не предложил правку"
                _emit(r, {"kind": "info", "text": r["note"]})
                break
            if outcome == "invalid":
                _emit(r, {"kind": "rejected", "text": x})
                prev = x
                continue
            r["versions"].append(x)
            _emit(r, {"kind": "evaluating", "version": x})
            v = await asyncio.to_thread(evaluate, x)
            ok = v["status"] == "candidate"
            _emit(r, {"kind": "evaluated", "version": x, "status": v["status"], "passed": ok, "base": _brief(parent),
                      "reasons": (v.get("gate") or {}).get("reasons", []) or ([v["error"][:300]] if v.get("error") else []),
                      "metrics": _brief(v)})
            if ok:
                r["accepted"].append(x)
                r["best"], parent = x, v
            prev = v
        r["status"] = "done"
    except _BudgetOut:
        r.update(status="done", note=f"бюджет ${r['budget_usd']:g} исчерпан посреди шага {r['step']} (${r['cost_usd']:.2f}) — незаконченная правка отброшена")
        _emit(r, {"kind": "info", "text": r["note"]})
    except asyncio.CancelledError:
        r["status"] = "cancelled"
        _emit(r, {"kind": "error", "message": "остановлено — агент убит"})
    except (_StepFailed, OSError, FileNotFoundError) as e:
        r.update(status="failed", error=str(e))
        _emit(r, {"kind": "error", "message": str(e)})
    r["finished_at"] = _now()


def _task_md(parent, task):
    m, g = parent.get("metrics") or {}, parent.get("gate") or {}
    brief = {k: m.get(k) for k in ("harsh0", "harsh50", "stress", "local", "negative_runs", "pilot_hit_rate", "diversity", "runtime_max")}
    return "\n".join([
        f"# Задача\n\n{task}\n",
        f"## Версия-родитель {parent['id']} ({parent['status']})\n\n"
        f"История, не задача: родитель уже содержит правку «{parent.get('hypothesis') or '—'}», повторять её не нужно.\n",
        "Главный критерий приёмки (gate): медиана и худший мир harsh0 не хуже родителя; stress и harsh50 — не хуже чем на 3%;",
        "must-have тесты: " + ", ".join(MUST) + ".\n",
        "## Метрики родителя (net в у.е.)\n\n```json\n" + json.dumps(brief, ensure_ascii=False, indent=1) + "\n```\n",
        "## Issues родителя\n\n" + ("\n".join(f"- {i['code']} ({i['severity']}): {i['evidence']}" for i in parent.get("issues", [])) or "нет") + "\n",
        "## Gate родителя\n\n" + ("; ".join(g.get("reasons", [])) or ("пройден" if g.get("passed") else "не оценён")) + "\n",
        "Справка: `docs/agent.md` (алгоритм), `docs/experiments.md` (ручные эксперименты и почему отклонены), `docs/lab.md`.",
        AI_RULES,
        "## Журнал: что уже пробовали в лаборатории (новые внизу)\n\n" + (_journal() or "пока пусто") + "\n",
    ])


JOURNAL_MAX = 30


def _journal():
    """Все версии, кроме baseline, с вердиктом gate: агент видит отклонённые идеи и не ходит по кругу между запусками."""
    out = []
    for v in [v for v in versions() if v["kind"] != "baseline" and v["status"] in ("candidate", "failed", "promoted")][-JOURNAL_MAX:]:
        what = (v.get("rationale") if v["kind"] == "ai" else v.get("hypothesis")) or ", ".join(d["key"] for d in v["diff"]) or "—"
        g = v.get("gate") or {}
        verdict = "принята" if v["status"] != "failed" else "отклонена: " + ("; ".join(g.get("reasons", [])) or (v.get("error") or "")[:150])
        diff = ", ".join(f"{d['key']} {d['from']}→{d['to']}" for d in v["diff"])
        out.append(f"- {v['id']} от {v['parent_id']} ({v['kind']}{'; ' + diff if diff else ''}): {' '.join(what.split())[:300]} → {verdict}")
    return "\n".join(out)


def _sandbox(parent, task):
    box = Path(tempfile.mkdtemp(prefix="cockpit-ai-"))
    (box / "agent.py").write_text(effective_source(parent))
    (box / "agent_parent.py").write_text(effective_source(parent))  # исходник: откатить, если правка не прошла lab_check
    m = parent.get("metrics") or {}
    (box / "lab_check.json").write_text(json.dumps({
        "parent_id": parent["id"], "seeds": SEEDS, "primary_seeds": PRIMARY_SEEDS, "tol": TH["gate_tol"], "runtime": TH["runtime"],
        "workers": WORKERS, "parent": {k: {x: m[k].get(x) for x in ("median", "min")} for k in ("harsh0", "harsh50", "stress") if k in m},
        "note": "родитель оценён с LLM-экспертом, в песочнице LLM нет — сравнение приблизительное" if m.get("llm_used") else None}))
    for f in SANDBOX_FILES:
        if (ROOT / f).exists():
            shutil.copy(ROOT / f, box / f)
    (box / "docs").mkdir()
    for f in SANDBOX_DOCS:
        if (ROOT / "docs" / f).exists():
            shutil.copy(ROOT / "docs" / f, box / "docs" / f)
    for src in [ROOT / "data", *ROOT.glob("*.csv")]:
        if src.name == "submission.csv" or not src.exists():
            continue
        try:
            (box / src.name).symlink_to(src.resolve())
        except OSError:  # Windows без прав на симлинки
            (shutil.copytree if src.is_dir() else shutil.copy)(src.resolve(), box / src.name)
    (box / "LAB_TASK.md").write_text(_task_md(parent, task))
    return box


if __name__ == "__main__":
    # validate: чужие ключи отсекаются, значения клипуются
    c, e = validate({"LCB_K": 9, "PILOT_SIZING": "wild", "evil": 1, "USE_LLM": "false"})
    assert c == {"LCB_K": 1.5, "USE_LLM": False} and len(e) == 2, (c, e)
    # константы: чтение/запись, '#' в строке и хвостовой комментарий переживают round-trip
    src = (ROOT / "agent.py").read_text()
    cfg = read_constants(src)
    new = set_constants(src, {"LCB_K": 0.75, "LLM_PROMPT_EXTRA": 'учти # и "кавычки"'})
    assert read_constants(new) == {**cfg, "LCB_K": 0.75, "LLM_PROMPT_EXTRA": 'учти # и "кавычки"'}
    assert "LCB_K = 0.75         # в план" in new and new.count("\n") == src.count("\n")
    # код версии: модуль из текста, DATA_DIR как у agent.py
    m = _load_agent(set_constants(src, {"LCB_K": 0.9}))
    assert m.LCB_K == 0.9 and m.DATA_DIR == agent.DATA_DIR and issubclass(_traced(m.Agent), m.Agent)
    # детекторы на синтетике
    run = lambda net, test="stress_10", **kw: {"test": test, "seed": 0, "net": net, "pilot_obs": [-0.1, -0.2, 0.1], "fallback": False,  # noqa: E731
                                                "crash": None, "overlap": 0.0, "planned": [{"mu": 0.01, "sd": 0.05, "S": 1, "n": 1, "ch": "call"}],
                                                "campaigns": [{"name": "c", "channel": "call", "target": "t", "n": 10, "gross": 1, "cost": 5}],
                                                "experts": {"llm": {"arms": 5, "pilots": 2, "planned": 0, "gain": 0}, "prior": {"gain": 1}}, **kw}
    codes = {i["code"] for i in detect([run(5e6), run(-1e6), run(6e6, fallback=True)])}
    assert {"HIGH_VARIANCE_BETWEEN_SEEDS", "TOO_MANY_NEGATIVE_PILOTS", "WEAK_LLM_CONTRIBUTION", "CALL_CHANNEL_NOT_COST_EFFECTIVE",
            "LOW_POSTERIOR_CONFIDENCE", "FALLBACK_TRIGGERED", "NO_DIVERSITY_IN_CAMPAIGNS"} <= codes, codes
    # gate
    t_ok = [{"name": n, "must": True, "passed": True} for n in MUST]
    mk = lambda h0, h50, st, lo=1.0: {"harsh0": {"median": h0, "min": lo}, "harsh50": {"median": h50}, "stress": {"median": st},  # noqa: E731
                                      "invalid": 0}
    base = {"id": "p", "metrics": mk(5.0, 6.0, 7.0)}
    # в жёстких лучше, в стресс-мирах −2% (в пределах допуска) — проходит
    assert gate({"tests": t_ok, "metrics": mk(5.5, 6.0, 6.86)}, base)["passed"]
    # главное семейство без допуска: −1% в harsh0 — отказ
    assert not gate({"tests": t_ok, "metrics": mk(4.95, 6.0, 7.0)}, base)["passed"]
    g = gate({"tests": t_ok[:-1] + [{**t_ok[-1], "passed": False}], "metrics": mk(5.0, 6.0, 6.0, lo=0.5)}, base)
    assert not g["passed"] and len(g["reasons"]) == 3, g
    # lab_check (локальный gate агента) решает так же, как gate
    import lab_check
    for cand in (mk(5.5, 6.0, 6.86), mk(4.95, 6.0, 7.0), mk(5.0, 6.0, 6.0, lo=0.5), mk(5.0, 5.7, 7.0)):
        assert bool(lab_check.verdict(cand, base["metrics"], TH["gate_tol"])) == (not gate({"tests": t_ok, "metrics": cand}, base)["passed"]), cand
    assert "перепрогони" in gate({"tests": t_ok, "metrics": mk(5, 6, 7)}, {"id": "old", "metrics": {"stress": {"median": 7}, "negative_runs": 0}})["reasons"][0]
    # быстрая матрица на baseline + один шаг ремедиации (без LLM)
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ.pop("OPENROUTER_API_KEY", None)
    db.use_schema("test_lab")  # воркеры пула наследуют схему через env
    db.drop_schema()
    db.init()
    SEEDS, PRIMARY_SEEDS = SEEDS[:2], PRIMARY_SEEDS[:2]
    ensure_baseline()
    v1 = evaluate("v001")
    assert v1["status"] == "promoted" and all(t["passed"] for t in v1["tests"] if t["must"]), [(t["name"], t["passed"], t["detail"]) for t in v1["tests"]]
    # …и считает мир так же, как матрица: net совпадает до копейки
    for key, test in (("harsh0", "harsh_0"), ("stress", "stress_10")):
        mine = lab_check.one((key, 0))[2]
        assert mine == next(r["net"] for r in v1["runs"] if r["test"] == test and r["seed"] == 0), (key, mine)
    assert "agent_parent.py" in os.listdir(box := _sandbox(v1, "t")) and json.loads((box / "lab_check.json").read_text())["parent"]["harsh0"]
    shutil.rmtree(box)
    made = remediate("v001", steps=1)
    v2 = load(made[0])
    assert v2["parent_id"] == "v001" and v2["diff"] and v2["status"] in ("candidate", "failed"), v2["status"]
    # AI-версия: константы из её кода, дети наследуют код, матрица идёт на этом коде
    ai_src = set_constants(src, {"LCB_K": 0.9}) + "\n# ai-edit\n"
    v3 = create("v001", {}, created_by="ai", kind="ai", source=ai_src)
    v4 = create(v3["id"], {"PILOT_MAX": 150})
    assert v3["config"]["LCB_K"] == 0.9 and v3["diff"] == [{"key": "LCB_K", "from": v1["config"]["LCB_K"], "to": 0.9}]
    assert v4["source_sha"] == v3["source_sha"] and "+# ai-edit" in code_diff(v3["id"]) and "PILOT_MAX = 150" in effective_source(v4)
    v3 = evaluate(v3["id"])
    assert v3["tests"] and not v3.get("error"), v3.get("error")
    # автономный цикл на фейковом CLI: хорошая правка → сломанная → «нечего улучшать»
    seen = []

    async def fake_run(h, model, prompt, cwd, timeout, on_start=None, budget_usd=None):
        task = (cwd / "LAB_TASK.md").read_text()
        seen.append(task)
        code = (cwd / "agent.py").read_text()
        edit = [lambda c: set_constants(c, {"PILOT_MAX": 150}), lambda c: c + "\ndef broken(:\n", lambda c: c][len(seen) - 1]
        (cwd / "agent.py").write_text(edit(code))
        yield {"kind": "tool", "tool": "Edit", "detail": f"{cwd}/agent.py"}
        yield {"kind": "result", "ok": True, "text": f"шаг {len(seen)}", "cost_usd": 0.01, "tokens_in": 100, "tokens_out": 10}
    harness.run = fake_run
    r = ai_start("v001", "claude", "", "подними PILOT_MAX", "t@t", steps=5, budget_usd=1)
    while ai_busy():
        time.sleep(0.2)
    r = run_load(r["id"])
    kinds = [e["kind"] for e in r["events"]]
    assert r["status"] == "done" and len(r["versions"]) == 1 and len(seen) == 3, (r["status"], r["error"], r["versions"], kinds)
    assert "rejected" in kinds and kinds[-1] == "info" and abs(r["cost_usd"] - 0.03) < 1e-9 and r["tokens_in"] == 300, (kinds, r)
    assert "не прошла проверку кода" in seen[2] and ("Прошлый шаг принят" in seen[1] or "отклонена" in seen[1]), seen[1][:300]
    va = load(r["versions"][0])
    assert va["ai"]["run_id"] == r["id"] and va["ai"]["cost_usd"] == 0.01 and va["tests"] and "Edit agent.py" in va["ai"]["log"]
    assert not any("cockpit-ai-" in (e.get("detail") or "") for e in r["events"])
    # перезапуск сервера посреди запуска: сирота-CLI убит, песочница удалена, запуск помечен failed
    box = Path(tempfile.mkdtemp(prefix="cockpit-ai-"))
    orphan = subprocess.Popen(["bash", "-c", "exec -a claude sleep 300"])
    time.sleep(0.2)
    assert harness.alive_cli(orphan.pid)
    run_save({**r, "id": "r999", "status": "running", "pid": orphan.pid, "box": str(box)})
    ai_recover()
    assert orphan.wait(timeout=5) == -9 and not box.exists() and run_load("r999")["status"] == "failed"
    print(f"ok: v001 harsh0 {_m(v1['metrics']['harsh0']['median'])}, stress {_m(v1['metrics']['stress']['median'])}, issues {[i['code'] for i in v1['issues']]}; "
          f"{v2['id']} ({v2['hypothesis']}) → {v2['status']} {v2['gate']['reasons']}")
    pool().shutdown()
    db.drop_schema()
