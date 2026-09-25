"""
Агент тарифных кампаний: эксперты предлагают рукава → адаптивные пилоты (EI) → жадный план.

Ключевое наблюдение (из описания среды): эффект зависит только от ячейки
(current_tariff, arpu_segment), целевого тарифа и канала, причём канал лишь
умножает эффект. Поэтому гипотеза = ячейка × target, а один SMS-пилот даёт
оценку для всех каналов сразу (base = observed / multiplier).
"""

import functools
import hashlib
import io
import json
import math
import os
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

NOISE_STD = 0.804   # шум на абонента, опубликован в документации среды
PRIOR_STD = 0.25    # широкий: в боевой среде эффекты другие, чем в истории
ARMS_PER_CELL = 4
ARMS_BIG_FRAC = 0.10 # ячейка с долей ARPU ≥ порога получает все переходы из истории (0 = выключено)
LCB_K = 0.5         # в план: mu - k*sigma > 0; 0 уходит в минус в пессимистичных мирах, 1 слишком робок
EI_STOP = 0.005     # хватит разведки, когда EI < 0.5% от стартового максимума
PILOT_CH = "sms"    # в 1.69 раза информативнее push на контакт
USE_FIT = True      # поправка prior: влезает ли трафик аудитории в пакет target
# CatBoost-prior: pct абонента = f(ARPU, трафик, from, to, пакеты), усреднённый по строкам АУДИТОРИИ ячейки.
# "hist" = prior только из истории (как было); модель включает версия lab.py, прошедшая gate.
PRIOR_MODEL = "hist"        # "catboost_shift": модель сдвигает групповое среднее; "catboost_full": смесь модель/история
PRIOR_ALPHA_MAX = 0.7       # full: вес модели при большой истории перехода
PRIOR_ALPHA_K = 30          # full: α(n) = ALPHA_MAX·n/(n+K) — при малом n не верим и модели
RISK_LAMBDA = 0.0           # штраф μ −= λ·u, u = 1/√(n+1) + доля строк вне диапазона train
PRIOR_WEIGHT = "rows"       # "arpu": среднее предсказаний с весом predicted_arpu
SUPPORT_MIN = 10            # full: переходов меньше — модель может только понизить μ
DATA_DIR = Path(__file__).parent / "data"
KEY = ["tariff_plan_code_from", "seg", "tariff_plan_code_to"]
LLM_PER_CELL = 2
TIME_LIMIT = 480     # с; ТЗ даёт 10 минут на act, 2 минуты запаса на план и fallback
LLM_CLIP = 0.5      # |base| из истории почти всегда < 0.35; больше — фантазия модели
PILOT_FRAC = 0.08   # размер пилота — доля ячейки в пределах [PILOT_MIN, PILOT_MAX]
PILOT_MIN = 60
PILOT_MAX = 200
# Флаги ниже по умолчанию выключены (= поведение, проверенное в docs/experiments.md), кроме PILOT_SIZING
# и схемы «prior по доле + совместная калибровка + MILP» (PRIOR_SD, JOINT, PLANNER, PUSH_FILL).
# Включает их версия из lab.py, только если прошла benchmark gate на стресс-мирах.
PILOT_SIZING = "adaptive"   # большой пилот только спорным рукавам, уверенные не перепроверяем; "fixed" — всем PILOT_FRAC
PILOT_BUDGET_SHARE = 1.0    # доля бюджета, которую можно потратить на пилоты
PILOT_CONTACT_SHARE = 1.0   # доля охвата, которую можно потратить на пилоты
PILOT_VALUE = "voi"         # "ei": expected improvement; "voi": knowledge gradient − цена пилота (с учётом выгоды самого пилота)
LCB_K_PILOT = 0.5           # k для рукавов после пилота (у непилотированных остаётся LCB_K)
PRIOR_SD = "share"          # "flat": PRIOR_STD всем; "share": sd рукава = √(SD0² + (SD_SHARE·доля перехода)²) вместо PRIOR_STD
SD0 = 0.03
SD_SHARE = 1.0
JOINT = True                # совместный апостериор с калибровкой θ = β0 + β1·m0 (см. _joint)
B0, B1 = 0.03, 0.5          # sd общего сдвига и наклона доверия к истории
SCALE_AT = "end"            # "each": после каждого пилота (сужает prior рано — хуже keep0.5/keep0c); "end": масштаб подбирается один раз перед планом, разведка идёт с широким prior
SCALE_MODE = "bma"          # "ml": одна лучшая точка сетки (ошибается, когда пилоты только на частых переходах); "bma": не одна точка сетки масштабов, а усреднение по всей сетке с весами правдоподобия
SCALE_MIN_OBS = 3           # с какого числа пилотов подбирать масштаб prior по правдоподобию
SCALE_GRID_A = (0.5, 1.0, 2.0, 4.0, 8.0)
SCALE_GRID_B = (0.25, 0.5, 1.0, 1.5, 2.0)
PLANNER = "milp"            # "greedy": прежний жадный план; "milp": точный отбор ячеек/target/каналов под все лимиты (scipy), при сбое — greedy
MILP_TIME = 20              # с на решатель
PUSH_FILL = True            # milp: ячейки с μ > 0, но LCB ≤ 0 могут идти в план бесплатным каналом
ADAPT = True                # JOINT: после ADAPT_AFTER пилотов проверить, надёжна ли история, и сменить режим пилотов
ADAPT_AFTER = 8
ADAPT_MODE = "exploit"      # "exploit": оставшиеся пилоты — полного размера на лучших кандидатах плана; "stop": разведка заканчивается
LLM_GATE = 1.0              # LLM-рукав только в ячейках, где лучший prior mu < порога; 1.0 = всегда (prior < 0.35)
RANK_BY = "mu"              # "lcb": target в ячейке и очередь охвата по mu - LCB_K*sigma
CHANNEL_MU = "mu"           # "lcb": апгрейд канала по консервативной оценке — дорогой канал слабым ячейкам не достаётся
UPGRADE_MIN_ROI = 0.0       # апгрейд канала только при dnet/dcost >= порога
META_CONTROLLER = False     # вес LLM = hit rate его пилотов; непроверенные догадки LLM умножаются на вес
USE_LLM = True
LLM_PROMPT_EXTRA = ""       # patch target lab.py: дописывается к инструкции LLM
# privacy gateway: в LLM уходят только агрегаты по ячейкам из allowlist, сырые строки — никогда
PRIVACY_MODE = "aggregate_only"   # synthetic_only: зашумлённые медианы и n; debug_safe: ни одного числа об абонентах
K_MIN = 10                  # ячейки меньше K_MIN идут без статистик
LLM_FIELDS = ("ARPU_3m_avg", "DATA_VOLUME", "OUT_LOC_OFFNET_MIN")
LLM_INSTRUCTION = (
    "Ты аналитик телеком-оператора. Нужно выбрать, на какой тариф предлагать перейти "
    "абонентам каждой ячейки (текущий тариф × ARPU-сегмент), чтобы вырос ARPU.\n"
    "В контексте: tariffs_csv — справочник тарифов; cells — ячейки: cur, seg, n (число абонентов) и медианы "
    "ARPU_3m_avg, DATA_VOLUME (МБ), OUT_LOC_OFFNET_MIN (минуты на других операторов), если ячейка не слишком мала.\n"
    "Для каждой ячейки предложи до {per_cell} target-тарифов (не равных текущему) и оценку "
    "expected — ожидаемое относительное изменение ARPU всей ячейки с учётом того, что перейдёт лишь "
    "часть абонентов (типичные значения от -0.1 до 0.3). Ответ строго JSON: "
    '{{"arms": [{{"cur": "...", "seg": "...", "target": "...", "expected": 0.05}}]}}'
)


def _history(data_dir=None):
    data_dir = data_dir or DATA_DIR  # при вызове: server подменяет DATA_DIR на загруженные выгрузки
    h = pd.read_csv(data_dir / "change_tariff.csv")
    h = h[h["AVG_ARPU_PREV_3M"] >= 100].copy()
    h["seg"] = pd.cut(h["AVG_ARPU_PREV_3M"], [-np.inf, 1000, 5000, np.inf],
                      labels=["LOW", "MID", "HIGH"]).astype(str)
    h["pct"] = ((h["AVG_ARPU_NEXT_3M"] - h["AVG_ARPU_PREV_3M"]) / h["AVG_ARPU_PREV_3M"]).clip(-1, 3)
    return h


def _fit_shift(h, profile, dict_tariff, data_dir=None):
    """
    Сдвиг mean pct по (from, seg, to): β × (доля аудитории ячейки, чей трафик влезает
    в пакет target − та же доля в истории перехода). β — внутри-переходная регрессия
    pct на «влезает», т.е. не путается с тем, какие переходы вообще популярны.
    """
    pkg = dict_tariff.set_index("tariff_plan_code")["Data_in_PKG"]
    usage = (pd.read_csv((data_dir or DATA_DIR) / "traffic.csv", usecols=["ID_NUMBER", "DATA_VOLUME"])
             .groupby("ID_NUMBER")["DATA_VOLUME"].mean())
    h = h[h["ID_NUMBER"].isin(usage.index)].copy()
    h["fit"] = (h["ID_NUMBER"].map(usage) <= h["tariff_plan_code_to"].map(pkg)).astype(float)
    grp = h.groupby(KEY)
    d_fit, d_pct = h["fit"] - grp["fit"].transform("mean"), h["pct"] - grp["pct"].transform("mean")
    beta = float((d_fit * d_pct).sum() / (d_fit ** 2).sum())
    f_hist = grp["fit"].mean()

    dv = profile["DATA_VOLUME"]
    f_aud = {}
    for t, cap in pkg.items():
        fits = (dv <= cap).where(dv.notna()).astype(float)
        for (cur, seg), v in fits.groupby([profile["current_tariff"], profile["arpu_segment"]]).mean().items():
            f_aud[(cur, seg, t)] = v
    shift = {k: beta * (f_aud[k] - fh) for k, fh in f_hist.items() if k in f_aud and not np.isnan(f_aud[k])}
    return pd.Series(shift, dtype=float), beta


def _prior(h, shift=None, mean=None, share=None):
    """
    base = shrunk (mean pct + сдвиг) × conversion share по (from, seg, to).
    mean (catboost_full) — готовый μ по KEY, заменяет shrunk mean; conversion share остаётся.
    share (dict) — заполняется долей перехода в ячейке: от неё зависит, насколько велик может быть эффект.
    """
    g = h.groupby(KEY)["pct"].agg(["mean", "size"])
    if shift is not None:
        g["mean"] = g["mean"] + shift.reindex(g.index).fillna(0.0)
    mu = g["mean"] * g["size"] / (g["size"] + 10)
    if mean is not None:
        mu = mean.reindex(g.index).fillna(mu)
    g["mu"] = mu
    g = g.reset_index()
    g["total"] = g.groupby(["tariff_plan_code_from", "seg"])["size"].transform("sum")
    g["base"] = g["mu"] * g["size"] / g["total"]
    if share is not None:
        share.update({(r.tariff_plan_code_from, r.seg, r.tariff_plan_code_to): r.size / r.total for r in g.itertuples()})
    return {(r.tariff_plan_code_from, r.seg, r.tariff_plan_code_to): r.base for r in g.itertuples()}


# --- CatBoost-prior ----------------------------------------------------------
CAT = ["tariff_plan_code_from", "tariff_plan_code_to", "seg", "key"]
PKG = {"price_tariff": "price", "Data_in_PKG": "data", "Min_another_operator_in_PKG": "min",
       "Min_another_operator_and_city_in_PKG": "min_city"}


def usage_cols(profile_columns, data_dir=None):
    """Потребление, которое есть и в профиле аудитории, и в traffic.csv истории (имена и единицы совпадают)."""
    t = pd.read_csv((data_dir or DATA_DIR) / "traffic.csv", nrows=0).columns
    return tuple(sorted((set(t) & set(profile_columns)) - {"ID_NUMBER", "tariff_plan_code"}))


def _usage(cols, data_dir=None):
    """Трафик истории: среднее за 3 последних месяца до перехода — как 3m-средние в профиле."""
    t = pd.read_csv((data_dir or DATA_DIR) / "traffic.csv", usecols=["ID_NUMBER", "time_key", *cols])
    t = t[t["time_key"].isin(sorted(t["time_key"].unique())[-3:])]
    return t.groupby("ID_NUMBER")[list(cols)].mean()


def hist_rows(h, usage):
    """Строки истории в формате _features: from/seg/to, arpu, вес, трафик, pct."""
    r = h[["ID_NUMBER", *KEY, "pct"]].assign(arpu=h["AVG_ARPU_PREV_3M"], w=h["AVG_ARPU_PREV_3M"])
    return r.join(usage, on="ID_NUMBER")


def profile_rows(profile, cols):
    """Строки аудитории в формате _features (без target). ARPU < 100 отброшен, как в _history: модель там не училась."""
    profile = profile[profile["ARPU_3m_avg"] >= 100]
    return pd.DataFrame({"tariff_plan_code_from": profile["current_tariff"], "seg": profile["arpu_segment"].astype(str),
                         "arpu": profile["ARPU_3m_avg"], "w": profile["predicted_arpu"], **{c: profile[c] for c in cols}})


def _features(rows, cols, tariffs):
    """Одна функция для обучения и инференса: одинаковые колонки, порядок и единицы."""
    X = rows[["tariff_plan_code_from", "tariff_plan_code_to", "seg", "arpu", *cols]].copy()
    X["key"] = X["tariff_plan_code_from"] + "|" + X["seg"] + "|" + X["tariff_plan_code_to"]
    t = tariffs.set_index("tariff_plan_code").astype({c: float for c in PKG})  # float и в train (есть NaN), и в inference
    for c, name in PKG.items():
        X[f"{name}_from"] = X["tariff_plan_code_from"].map(t[c])
        X[f"{name}_to"] = X["tariff_plan_code_to"].map(t[c])
        X[f"d_{name}"] = X[f"{name}_to"] - X[f"{name}_from"]
    dv = X["DATA_VOLUME"]
    X["fit_from"] = (dv <= X["data_from"]).where(dv.notna()).astype(float)
    X["fit_to"] = (dv <= X["data_to"]).where(dv.notna()).astype(float)
    X["use_to"] = dv / X["data_to"].clip(lower=1)
    X["over_to"] = X["data_to"] / dv.clip(lower=1)
    return X


def fit_prior_model(rows, cols, tariffs):
    """CatBoost на pct: (model, min, max признаков train) — диапазон для OOD-прокси."""
    from catboost import CatBoostRegressor  # ponytail: импорт здесь — без catboost агент работает в режиме hist
    X = _features(rows, cols, tariffs)
    m = CatBoostRegressor(loss_function="RMSE", iterations=500, learning_rate=0.05, depth=6, l2_leaf_reg=8,
                          random_seed=42, verbose=0, thread_count=4, allow_writing_files=False)
    m.fit(X, rows["pct"], cat_features=CAT)
    num = X.drop(columns=CAT)
    return m, num.min(), num.max()


@functools.lru_cache(maxsize=1)
def _train_prior_model(tariffs_csv, cols):
    """История статична: обучаем раз на процесс (stress/lab зовут act десятки раз)."""
    return fit_prior_model(hist_rows(_history(), _usage(cols)), cols, pd.read_csv(io.StringIO(tariffs_csv)))


def model_prior(fit, rows, keys, cols, tariffs, weight=None):
    """
    pct для строк × target из keys (только переходы из истории их ячейки), агрегат по KEY:
    model_mean (среднее по строкам или с весом ARPU), ood — доля строк вне диапазона train,
    worse_fit — target дороже и в среднем хуже по fit, чем текущий.
    """
    model, lo, hi = fit
    x = rows.drop(columns=["tariff_plan_code_to", "pct"], errors="ignore").merge(
        pd.DataFrame(list(keys), columns=KEY), on=KEY[:2])
    X = _features(x, cols, tariffs)
    num = X.drop(columns=CAT)
    x["p"] = model.predict(X)
    x["ood"] = ((num < lo) | (num > hi)).any(axis=1).astype(float)
    x["w"] = x["w"].clip(lower=1.0) if (weight or PRIOR_WEIGHT) == "arpu" else 1.0
    x["pw"], x["d_price"], x["fit_from"], x["fit_to"] = x["p"] * x["w"], X["d_price"], X["fit_from"], X["fit_to"]
    g = x.groupby(KEY)
    out = g[["pw", "w"]].sum()
    out = pd.DataFrame({"model_mean": out["pw"] / out["w"], "ood": g["ood"].mean(), "d_price": g["d_price"].first()})
    out["worse_fit"] = (out["d_price"] > 0) & (g["fit_to"].mean() < g["fit_from"].mean())
    return out


def model_mu(h, mp, mode=None):
    """
    (shift, mean, диагностика по KEY) для _prior.
    shift:  model_mean − hist_mean − λ·u (дальше обычное сжатие n/(n+10)).
    full:   μ = α(n)·model + (1−α)·hist_s − λ·u, hist_s = hist_mean·n/(n+10);
            guardrails (мало истории, экстраполяция, дороже и хуже по fit) — модель только понижает μ.
    """
    mode = mode or PRIOR_MODEL
    g = h.groupby(KEY)["pct"].agg(["mean", "size"]).join(mp, how="inner")
    n = g["size"]
    g["u"] = 1 / np.sqrt(n + 1) + g["ood"]
    g["hist_s"] = g["mean"] * n / (n + 10)
    if mode == "catboost_shift":
        shift = g["model_mean"] - g["mean"] - RISK_LAMBDA * g["u"]
        g["mu"], g["guard"] = (g["mean"] + shift) * n / (n + 10), False
        return shift, None, g
    a = PRIOR_ALPHA_MAX * n / (n + PRIOR_ALPHA_K)
    mu = a * g["model_mean"] + (1 - a) * g["hist_s"] - RISK_LAMBDA * g["u"]
    g["guard"] = (n < SUPPORT_MIN) | (g["ood"] > 0.5) | g["worse_fit"]
    g["mu"] = mu.where(~g["guard"], np.minimum(mu, g["hist_s"])).clip(-1, 3)
    return None, g["mu"], g


def llm_config(model=None):
    """(base_url, key, model): OpenRouter, если есть OPENROUTER_API_KEY, иначе OpenAI."""
    if os.environ.get("OPENROUTER_API_KEY"):
        return ("https://openrouter.ai/api/v1", os.environ["OPENROUTER_API_KEY"],
                model or os.environ.get("LLM_MODEL", "openai/gpt-4o-mini"))
    return (os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"), os.environ.get("OPENAI_API_KEY"),
            model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))


def _llm_call(prompt, model=None):
    base, key, model = llm_config(model)
    body = {"model": model, "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": prompt}]}
    # reasoning-модели (gpt-5*, gpt-6*, o*) отвечают 400 на temperature≠1
    if not model.split("/")[-1].startswith(("gpt-5", "gpt-6", "o1", "o3", "o4")):
        body["temperature"] = 0
    req = urllib.request.Request(
        base + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def safe_context(profile, cells, tariffs_csv, mode=None):
    """
    Privacy gateway, шаг «агрегация/редакция»: raw профиль → safe JSON.
    Только ячейки и медианы полей из LLM_FIELDS; малые ячейки без статистик.
    Возвращает (ctx, redacted_fields) — второе для аудита.
    """
    mode = mode or PRIVACY_MODE
    med = profile.groupby(["current_tariff", "arpu_segment"])[list(LLM_FIELDS)].median()
    rng = np.random.default_rng(0)  # ponytail: шум synthetic_only детерминирован — агент воспроизводим
    rows = []
    for (cur, seg), c in cells.items():
        row = {"cur": cur, "seg": seg}
        if mode != "debug_safe":
            row["n"] = max(50, int(round(c["n"] / 50) * 50)) if mode == "synthetic_only" else int(c["n"])
            if c["n"] >= K_MIN:
                m = med.loc[(cur, seg)]
                if mode == "synthetic_only":
                    m = m * rng.uniform(0.8, 1.2, len(m))
                row.update({f: round(float(v)) for f, v in m.items() if pd.notna(v)})
        rows.append(row)
    return {"tariffs_csv": tariffs_csv, "cells": rows}, redacted_fields(profile.columns)


def redacted_fields(columns):
    """Поля профиля, которые не уходят в LLM ни в каком виде (ключи ячейки идут как cur/seg)."""
    return sorted(set(columns) - set(LLM_FIELDS) - {"current_tariff", "arpu_segment"})


def _keys(obj):
    if isinstance(obj, dict):
        return set(obj) | {k for v in obj.values() for k in _keys(v)}
    if isinstance(obj, list):
        return {k for v in obj for k in _keys(v)}
    return set()


def llm_proxy(task, ctx, instruction, audit, allowed, redacted=(), model=None):
    """
    Privacy gateway, шаг «запрос»: промпт строится только из instruction + safe JSON.
    Ключ вне allowlist → отказ до вызова. Каждый вызов (и сбой) пишется в audit.
    """
    extra = set(_keys(ctx)) - set(allowed)
    if extra:
        raise ValueError(f"privacy: поля вне allowlist: {sorted(extra)}")
    prompt = instruction + "\n\nКонтекст (JSON):\n" + json.dumps(ctx, ensure_ascii=False)
    rec = {"task": task, "model": llm_config(model)[2], "mode": PRIVACY_MODE, "fields_sent": sorted(_keys(ctx)), "redacted_fields": list(redacted),
           "prompt": prompt, "prompt_sha": hashlib.sha256(prompt.encode()).hexdigest()[:12],
           "response": None, "error": None, "latency": None, "parsed": [], "rejected": []}
    audit.append(rec)
    t = time.time()
    try:
        rec["response"] = _llm_call(prompt, model)
        return rec["response"]
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
        raise
    finally:
        rec["latency"] = round(time.time() - t, 2)


def _update(a, obs, n, mult):
    """Байесовское обновление рукава наблюдением base = obs на n абонентах: шум NOISE_STD/mult на абонента."""
    noise_var = (NOISE_STD / mult) ** 2 / n
    prec = 1 / a["var"] + 1 / noise_var
    a["mu"] = (a["mu"] / a["var"] + obs / noise_var) / prec
    a["var"], a["n"] = 1 / prec, a["n"] + n
    a.setdefault("obs", []).append(obs)
    a.setdefault("nv", []).append(noise_var)
    return obs


def _joint(arms, fit=True):
    """
    Совместный апостериор всех рукавов (JOINT): θ_k = β0 + β1·m0_k + u_k, β ~ N([0, 1], diag(B0², B1²)),
    u_k ~ N(0, a²·v0a_k + b²·v0b_k). Пилот любого рукава сдвигает β, а через него — все рукава: если история
    врёт, β1 → 0 и непилотированные рукава перестают наследовать её μ. Масштаб (a, b) — максимум правдоподобия
    наблюдений на сетке: пилоты сами решают, насколько широк prior и растёт ли он с долей перехода.
    """
    keys = list(arms)
    A = [arms[k] for k in keys]
    m = np.array([a["m0"] for a in A])
    va, vb = np.array([a["v0a"] for a in A]), np.array([a["v0b"] for a in A])
    J = np.array([i for i, a in enumerate(A) for _ in a.get("obs", ())], dtype=int)
    y = np.array([o for a in A for o in a.get("obs", ())])
    r = np.array([v for a in A for v in a.get("nv", ())])
    X = np.c_[np.ones(len(A)), m]
    Sb = np.diag([B0 ** 2, B1 ** 2])
    XS = X @ Sb
    same = (J[:, None] == J[None, :]) if len(J) else None

    def cov_obs(sa, sb):
        tau = sa ** 2 * va + sb ** 2 * vb
        S = XS[J] @ X[J].T + same * tau[J][:, None] + np.diag(r)
        return tau, S

    def post(sa, sb):
        """Апостериор рукавов и log-правдоподобие пилотов при масштабе (sa, sb)."""
        tau = sa ** 2 * va + sb ** 2 * vb
        mean, var, ll = m.copy(), (XS * X).sum(1) + tau, 0.0
        if len(J):
            _, S = cov_obs(sa, sb)
            d = y - m[J]
            CKJ = XS @ X[J].T + (np.arange(len(A))[:, None] == J[None, :]) * tau[:, None]
            mean = m + CKJ @ np.linalg.solve(S, d)
            var = var - (CKJ * np.linalg.solve(S, CKJ.T).T).sum(1)
            ll = -0.5 * (d @ np.linalg.solve(S, d) + np.linalg.slogdet(S)[1])
        return mean, var, ll

    grid = [(ga, gb) for ga in SCALE_GRID_A for gb in (SCALE_GRID_B if vb.any() else (1.0,))]
    if fit and SCALE_MODE == "bma":
        # байесовское усреднение по сетке масштабов (равный prior): пока пилотов мало, вариант «эффект не зависит
        # от доли перехода» сохраняет вес, и редкие переходы получают честную неопределённость, а не почти нулевую
        res = [post(ga, gb) for ga, gb in grid]
        lls = np.array([r[2] for r in res])
        w = np.exp(lls - lls.max()); w /= w.sum()
        mean = sum(wi * r[0] for wi, r in zip(w, res)) if len(J) else m.copy()  # без пилотов μ = история, без шума float
        var = sum(wi * (r[1] + r[0] ** 2) for wi, r in zip(w, res)) - mean ** 2
        sa, sb = grid[int(np.argmax(w))]
    else:
        sa, sb = 1.0, 1.0
        if fit and len(J) >= SCALE_MIN_OBS:  # ponytail: сетка 5×5 и полный перебор — q ≤ 20 наблюдений, это миллисекунды
            sa, sb = grid[int(np.argmax([post(ga, gb)[2] for ga, gb in grid]))]
        mean, var, _ = post(sa, sb)
    for a, mu, v in zip(A, mean, var):
        a["mu"], a["var"] = float(mu), float(max(v, 1e-8))
    return sa, sb


def _ei(mu, sd, best):
    z = (mu - best) / sd
    return sd * (z * 0.5 * (1 + math.erf(z / math.sqrt(2))) + math.exp(-z * z / 2) / math.sqrt(2 * math.pi))


def _lcb(a):
    return a["mu"] - (LCB_K_PILOT if a["n"] else LCB_K) * math.sqrt(a["var"])


class Agent:
    deadline = math.inf  # ставится в act; внутренние методы, вызванные напрямую (stress_eval), без лимита
    exploit = False      # ADAPT: история признана надёжной, пилоты — на лучших кандидатах плана
    model = None  # slug модели для LLM-эксперта; None → LLM_MODEL / OPENAI_MODEL из env
    # база знаний: прошлые пилоты и итоги кампаний на этой аудитории, dict(cur, seg, target, channel, n, lift_ratio).
    # Пусто без кокпита; server.py подкладывает строки из Postgres.
    feedback = ()

    def __init__(self):
        self.log, self.llm_audit, self.weights, self.share = [], [], {}, {}

    def act(self, env):
        self.log, self.llm_audit, self.weights, self.share = [], [], {}, {}
        self.exploit = False
        self.deadline = time.time() + TIME_LIMIT
        arms = {}
        try:
            cells, arms = self._arms(env)
            self._explore(env, cells, arms)
            if JOINT and SCALE_AT == "end":  # разведка — с широким prior, план — с масштабом, подобранным по пилотам
                self.scale = _joint(arms)
                self.log.append(f"joint: scale a={self.scale[0]}, b={self.scale[1]}")
            plan = self._plan(env, cells, arms)
            try:
                self._expert_report(arms)
            except Exception as e:  # отчёт — только лог, план из-за него не теряем
                self.log.append(f"report skipped: {type(e).__name__}: {e}")
            if plan:
                return plan
        except Exception as e:  # ponytail: любой сбой → безопасный fallback, а не падение
            self.log.append(f"fallback: {type(e).__name__}: {e}")
        return self._fallback(env, arms)

    # --- гипотезы: портфель экспертов ------------------------------------
    experts = ("prior", "llm")  # heuristic (upsell по пакету) пробовали — хуже минимум, см. docs/experiments.md

    def _arms(self, env):
        p = env.customer_profile
        cells = (p.groupby(["current_tariff", "arpu_segment"])
                 .agg(n=("ID_NUMBER", "size"), S=("predicted_arpu", "sum")))
        cells = {k: {"n": int(r.n), "S": float(r.S)} for k, r in cells.iterrows()}
        tariffs = list(env.tariffs["tariff_plan_code"])
        props = {}
        for name in self.experts:
            if name == "llm" and time.time() > self.deadline - 60:  # вызов LLM может занять до 60 с
                self.log.append("llm skipped: time limit")
                continue
            try:
                props[name] = getattr(self, f"_expert_{name}")(env, cells, tariffs)
                self.log.append(f"{name}: {len(props[name])} arms")
            except Exception as e:  # эксперт сломался — остальные работают
                self.log.append(f"{name} skipped: {type(e).__name__}: {e}")
        if "llm" in props:  # LLM закрывает пробелы prior, а не конкурирует с сильным prior за пилоты
            top = {}
            for k, mu in props.get("prior", {}).items():
                top[k[:2]] = max(top.get(k[:2], 0.0), mu)
            props["llm"] = {k: mu for k, mu in props["llm"].items() if top.get(k[:2], 0.0) < LLM_GATE}
        # реестр: один рукав на (cur, seg, target), src — кто его предложил;
        # mu — из истории, если она есть, иначе среднее догадок экспертов
        arms = {}
        for k in dict.fromkeys(k for pr in props.values() for k in pr):  # порядок важен: ничьи в EI
            src = {n for n, pr in props.items() if k in pr}
            mu = props["prior"][k] if "prior" in src else float(np.mean([props[n][k] for n in src]))
            arms[k] = {"mu": mu, "var": self._var0(k), "n": 0, "src": src, "obs": [], "m0": mu, **self._v0parts(k)}
        self._apply_feedback(env, cells, arms)
        return cells, arms

    def _var0(self, k):
        """Априорная дисперсия рукава: flat PRIOR_STD или растущая с долей перехода (PRIOR_SD="share")."""
        if PRIOR_SD == "share" and k in self.share:
            return SD0 ** 2 + (SD_SHARE * self.share[k]) ** 2
        return PRIOR_STD ** 2

    def _v0parts(self, k):
        """Априорная дисперсия рукава двумя слагаемыми (v0a — постоянная, v0b — от доли перехода) для JOINT."""
        if PRIOR_SD == "share" and k in self.share:
            return {"v0a": SD0 ** 2, "v0b": (SD_SHARE * self.share[k]) ** 2}
        return {"v0a": PRIOR_STD ** 2, "v0b": 0.0}

    def _apply_feedback(self, env, cells, arms):
        """Прошлые наблюдения на этой же аудитории — апдейт апостериора до разведки, как уже сделанные пилоты.
        Рукав, которого нет у экспертов, заводится с mu = 0: в план он попадёт по n > 0, как пилотированный."""
        # ponytail: без затухания — старый месяц весит как новый; FEEDBACK_DECAY, когда месяцев станет много
        used = set()
        for r in self.feedback:
            k = (r["cur"], r["seg"], r["target"])
            ch = env.channels.get(r["channel"])
            if k[:2] not in cells or k[0] == k[2] or not ch or not r["n"] > 0:
                continue
            a = arms.setdefault(k, {"mu": 0.0, "var": PRIOR_STD ** 2, "n": 0, "src": set(), "obs": [], "m0": 0.0,
                                    "v0a": PRIOR_STD ** 2, "v0b": 0.0})
            a["src"].add("feedback")
            _update(a, r["lift_ratio"] / ch["conversion_multiplier"], int(r["n"]), ch["conversion_multiplier"])
            used.add(k)
        if self.feedback:
            self.log.append(f"feedback: {len(self.feedback)} rows → {len(used)} arms")
        if JOINT:
            _joint(arms)

    def _expert_prior(self, env, cells, tariffs):
        prior = {}
        try:
            h = _history()
            shift = mean = None
            if PRIOR_MODEL != "hist":
                try:
                    shift, mean = self._model_prior(env, h)
                except Exception as e:  # нет catboost / traffic.csv — prior из истории, как в режиме hist
                    self.log.append(f"model skipped: {type(e).__name__}: {e}")
            if shift is None and mean is None and USE_FIT:
                try:
                    shift, beta = _fit_shift(h, env.customer_profile, env.tariffs)
                    self.log.append(f"fit: beta={beta:.3f}, pairs={len(shift)}")
                except Exception as e:  # нет traffic.csv и т.п. — prior без поправки
                    self.log.append(f"fit skipped: {type(e).__name__}: {e}")
            prior = _prior(h, shift, mean, self.share)
        except Exception as e:  # нет истории — рукава всё равно нужны, mu = 0
            self.log.append(f"history skipped: {type(e).__name__}: {e}")
        # сначала переходы из истории: иначе пара без истории (μ = 0) обгоняет переход с μ < 0 и съедает пилоты,
        # а в план не попадает (docs/experiments.md)
        out = {}
        total_s = sum(c["S"] for c in cells.values())
        for (cur, seg) in cells:
            # PRIOR_SD="share": ранг по потенциалу μ + σ — большой эффект возможен только у частого перехода
            up = (lambda k: prior.get(k, 0.0) + math.sqrt(self._var0(k))) if PRIOR_SD == "share" else (lambda k: prior.get(k, 0.0))
            ranked = sorted((t for t in tariffs if t != cur),
                            key=lambda t: ((cur, seg, t) in prior, up((cur, seg, t))), reverse=True)
            # крупная ячейка (≥ ARMS_BIG_FRAC всего ARPU) — все переходы из истории: ценность находки ∝ размеру ячейки,
            # а лучший target там бывает редким переходом, который в top-ARMS_PER_CELL не попадает
            big = ARMS_BIG_FRAC and cells[(cur, seg)]["S"] >= ARMS_BIG_FRAC * total_s
            k_arms = max(ARMS_PER_CELL, sum((cur, seg, t) in prior for t in tariffs)) if big else ARMS_PER_CELL
            for t in ranked[:k_arms]:
                out[(cur, seg, t)] = prior.get((cur, seg, t), 0.0)
        return out

    def _model_prior(self, env, h):
        p = env.customer_profile
        cols = usage_cols(p.columns)
        fit = _train_prior_model(env.tariffs.to_csv(index=False), cols)
        mp = model_prior(fit, profile_rows(p, cols), h.groupby(KEY).size().index, cols, env.tariffs)
        shift, mean, g = model_mu(h, mp)
        d = g["mu"] - g["hist_s"]
        self.log.append(f"model {PRIOR_MODEL}: arms={len(g)} |Δμ|>0.02: {(d.abs() > 0.02).sum()} "
                        f"mean|Δμ|={d.abs().mean():.3f} guard={int(g['guard'].sum())} weight={PRIOR_WEIGHT}")
        for k, r in g.nlargest(5, "u").iterrows():
            self.log.append(f"model uncertain {'/'.join(k)}: n={r['size']} ood={r.ood:.2f} u={r.u:.2f} "
                            f"hist={r.hist_s:.3f} model={r.model_mean:.3f} mu={r.mu:.3f}")
        self.prior_table = g  # для stress_eval --prior-debug и UI
        return shift, mean

    def _expert_llm(self, env, cells, tariffs):
        if not USE_LLM:
            self.log.append("llm: выключен (USE_LLM=False)")
            return {}
        if not llm_config()[1]:
            self.log.append("llm: нет OPENROUTER_API_KEY / OPENAI_API_KEY")
            return {}
        ctx, redacted = safe_context(env.customer_profile, cells, env.tariffs.to_csv(index=False))
        instruction = LLM_INSTRUCTION.format(per_cell=LLM_PER_CELL) + ("\n" + LLM_PROMPT_EXTRA if LLM_PROMPT_EXTRA else "")
        text = llm_proxy("expert_llm", ctx, instruction, self.llm_audit,
                         allowed={"tariffs_csv", "cells", "cur", "seg", "n", *LLM_FIELDS}, redacted=redacted, model=self.model)
        rec = self.llm_audit[-1]
        out = self._parse_llm(text, cells, tariffs, rec["rejected"], rec["parsed"])
        self.log.append(f"llm: prompt {rec['prompt_sha']} ({PRIVACY_MODE}), принято {len(out)}, "
                        f"отклонено {len(rec['rejected'])}, скрыто полей {len(redacted)}")
        return out

    @staticmethod
    def _parse_llm(text, cells, tariffs, rejected=None, parsed=None):
        """Privacy gateway, шаг «валидация ответа»: в реестр идут только существующие ячейки и тарифы."""
        rejected = [] if rejected is None else rejected
        parsed = [] if parsed is None else parsed
        try:
            rows = json.loads(text).get("arms", [])
        except (ValueError, AttributeError):
            rejected.append({"row": str(text)[:200], "reason": "bad_json"})
            raise
        out, per_cell = {}, {}
        for r in rows:
            try:
                k = (str(r["cur"]), str(r["seg"]), str(r["target"]))
                raw = float(r["expected"])
            except (KeyError, TypeError, ValueError):
                rejected.append({"row": r, "reason": "bad_fields"})
                continue
            reason = ("unknown_cell" if k[:2] not in cells else "unknown_tariff" if k[2] not in tariffs
                      else "same_as_current" if k[2] == k[0] else "per_cell_limit" if per_cell.get(k[:2], 0) >= LLM_PER_CELL
                      else None)
            if reason:
                rejected.append({"row": r, "reason": reason})
                continue
            out[k] = float(np.clip(raw, -LLM_CLIP, LLM_CLIP))
            per_cell[k[:2]] = per_cell.get(k[:2], 0) + 1
            parsed.append({"cur": k[0], "seg": k[1], "target": k[2], "expected": raw, "used": out[k],
                           "clipped": out[k] != raw})
        return out

    # --- адаптивные пилоты ----------------------------------------------
    @staticmethod
    def _best_other(arms, k):
        return max([0.0] + [b["mu"] for kk, b in arms.items() if kk[:2] == k[:2] and kk != k])

    def _arm_ei(self, cells, arms):
        return {k: cells[k[:2]]["S"] * _ei(a["mu"], math.sqrt(a["var"]), self._best_other(arms, k))
                for k, a in arms.items()}

    def _pilot_size(self, cells, arms, k):
        n = PILOT_FRAC * cells[k[:2]]["n"]
        if PILOT_SIZING == "adaptive":
            # |z| — насколько рукав уже явно лучше/хуже соседей: чем спорнее, тем больше выборка
            z = abs(arms[k]["mu"] - self._best_other(arms, k)) / math.sqrt(arms[k]["var"])
            n = min(PILOT_MIN + (PILOT_MAX - PILOT_MIN) / (1 + z), cells[k[:2]]["n"])
        return int(np.clip(n, PILOT_MIN, PILOT_MAX))

    @staticmethod
    def _marginal_value(cells, arms, room):
        """Ценность на контакт (без канала) у ячейки на отсечке охвата: её вытесняет каждый контакт пилота."""
        best = {}
        for k, a in arms.items():
            if _lcb(a) > 0 and a["mu"] > best.get(k[:2], 0.0):
                best[k[:2]] = a["mu"]
        for cell, mu in sorted(best.items(), key=lambda t: t[1] * cells[t[0]]["S"] / cells[t[0]]["n"], reverse=True):
            room -= cells[cell]["n"]
            if room < 0:
                return mu * cells[cell]["S"] / cells[cell]["n"]
        return 0.0  # ponytail: охвата хватает всем — контакты пилота бесплатны

    def _arm_net_voi(self, env, cells, arms, mult, cost):
        """
        Knowledge gradient − цена пилота. KG: насколько пилот на n сдвинет решение по ячейке
        (s̃ — sd сдвига μ); у уверенного лидера |μ − альтернатива| ≫ s̃ → ≈ 0, пилот не нужен.
        Цена: SMS + вытесненные из плана контакты − выигрыш самого пилота (он тоже скорится).
        """
        v_marg = mult * self._marginal_value(cells, arms, env.remaining_contacts)
        out = {}
        for k, a in arms.items():
            c, n = cells[k[:2]], self._pilot_size(cells, arms, k)
            s = a["var"] / math.sqrt(a["var"] + (NOISE_STD / mult) ** 2 / n)
            kg = c["S"] * mult * _ei(-abs(a["mu"] - self._best_other(arms, k)), s, 0.0)
            out[k] = kg - n * (cost + v_marg - mult * max(a["mu"], 0.0) * c["S"] / c["n"])
        return out

    @staticmethod
    def _refit(arms):
        """Апостериор по ходу разведки: bma — усреднение по сетке масштабов; ml — масштаб 1 или подбор после каждого пилота."""
        return _joint(arms, fit=SCALE_MODE == "bma" or SCALE_AT == "each")

    def _explore(self, env, cells, arms):
        mult = env.channels[PILOT_CH]["conversion_multiplier"]
        cost = env.channels[PILOT_CH]["cost_per_contact"]
        ei0, llm_hits = None, []
        total = getattr(env, "total_budget", env.remaining_budget)
        reach = getattr(env, "max_total_contacts", env.remaining_contacts)
        while env.pilots_left > 0:
            if time.time() > self.deadline:
                self.log.append("explore stopped: time limit")
                break
            voi = PILOT_VALUE == "voi"
            if self.exploit and ADAPT_MODE == "stop":
                self.log.append("explore stopped: история надёжна")
                break
            ei = self._arm_net_voi(env, cells, arms, mult, cost) if voi else self._arm_ei(cells, arms)
            if self.exploit:  # история надёжна: пилот = маленькая кампания на лучшем кандидате плана, польза = μ·Σ ARPU
                ei = {k: arms[k]["mu"] * cells[k[:2]]["S"] for k in ei if arms[k]["mu"] > 0 and self._eligible(arms[k], k)}
                if not ei:
                    break
            self._scores = ei
            if PILOT_SIZING == "adaptive":  # stop по posterior confidence: уверенный рукав не перепроверяем
                ei = {k: v for k, v in ei.items() if not arms[k]["n"]
                      or abs(arms[k]["mu"] - self._best_other(arms, k)) / math.sqrt(arms[k]["var"]) <= 2}
                if not ei:
                    break
            k = max(ei, key=ei.get)
            ei0 = ei0 or ei[k]
            if not self.exploit and ((ei[k] <= 0) if voi else (ei[k] < EI_STOP * ei0)):
                break
            cur, seg, target = k
            n = min(PILOT_MAX, cells[k[:2]]["n"]) if self.exploit else self._pilot_size(cells, arms, k)
            # политика: не больше PILOT_BUDGET_SHARE бюджета и PILOT_CONTACT_SHARE охвата на пилоты
            n = min(n, int((PILOT_BUDGET_SHARE * total - (total - env.remaining_budget)) / cost) if cost else n)
            n = min(n, int(PILOT_CONTACT_SHARE * reach - (reach - env.remaining_contacts)))
            if n < 10:
                break
            # резерв: на оставшиеся контакты должно хватить SMS в финальном плане
            if env.remaining_budget - n * cost < cost * (env.remaining_contacts - n):
                break
            try:
                res = env.run_pilot(target_tariff=target, channel=PILOT_CH, n_customers=n,
                                    filter_arpu_segment=seg, filter_current_tariff=cur)
            except (RuntimeError, ValueError):
                break
            a, m = arms[k], res["n_customers"]
            obs = _update(a, res["observed_lift_ratio"] / mult, m, mult)
            if JOINT:
                self.scale = self._refit(arms)
                if ADAPT and not self.exploit and sum(len(b.get("obs", ())) for b in arms.values()) >= ADAPT_AFTER:
                    sa, sb = _joint(arms)  # масштаб по пилотам: оба на минимуме сетки = пилоты укладываются в шум вокруг истории
                    self.exploit = sa <= min(SCALE_GRID_A) and sb <= min(SCALE_GRID_B)
                    if not self.exploit:
                        self.scale = self._refit(arms)
                    self.log.append(f"adapt: scale a={sa}, b={sb} → {'история надёжна, ' + ADAPT_MODE if self.exploit else 'разведка'}")
            self.log.append(f"pilot {cur}/{seg}->{target} n={m} obs={obs:.3f} post={a['mu']:.3f}±{math.sqrt(a['var']):.3f}")
            if META_CONTROLLER and a.get("src") == {"llm"}:
                # мета-контроллер: вес LLM = доля его пилотов в плюс; непроверенные догадки LLM сжимаются к 0
                llm_hits.append(obs > 0)
                w = self.weights["llm"] = float(np.mean(llm_hits))
                for b in arms.values():
                    if b.get("src") == {"llm"} and not b["n"]:
                        b["mu"] = b.setdefault("mu0", b["mu"]) * w
                self.log.append(f"meta: вес llm={w:.2f} после {len(llm_hits)} пилотов")

    # --- финальный план -------------------------------------------------
    def _plan(self, env, cells, arms):
        ch = env.channels
        names = sorted(ch, key=lambda c: ch[c]["cost_per_contact"])  # push, sms, ads, call
        rank = _lcb if RANK_BY == "lcb" else (lambda a: a["mu"])
        picks = []
        for (cur, seg), c in cells.items():
            # гипотеза без истории (только LLM) идёт в план лишь после пилота: догадке модели на слово не верим
            cand = [(rank(a), a["mu"], _lcb(a), math.sqrt(a["var"]), k[2]) for k, a in arms.items()
                    if k[:2] == (cur, seg) and ("prior" in a.get("src", ("prior",)) or a["n"])]
            if not cand:
                continue
            r, mu, lcb, sd, target = max(cand)
            if lcb > 0:
                picks.append({"cur": cur, "seg": seg, "target": target, "mu": mu, "sd": sd, "lcb": lcb, "rank": r, **c})

        val = (lambda x: x["lcb"]) if CHANNEL_MU == "lcb" else (lambda x: x["mu"])
        net = lambda x, c: val(x) * ch[c]["conversion_multiplier"] * x["S"] - ch[c]["cost_per_contact"] * x["n"]
        chosen = None
        if PLANNER == "milp":
            try:
                chosen = self._plan_milp(env, cells, arms, net)
            except Exception as e:  # нет scipy / решатель не справился — жадный план ниже
                self.log.append(f"milp skipped: {type(e).__name__}: {e}")
        if chosen is not None:
            self._chosen = chosen
        else:
            chosen = self._plan_greedy(env, picks, net, names)

        # группировка: arpu × список тарифов выбирает ровно нужные ячейки
        return self._pack(chosen, net)

    def _eligible(self, a, k):
        """Рукав может идти в план: есть история или пилот и LCB > 0."""
        return ("prior" in a.get("src", ("prior",)) or a["n"]) and _lcb(a) > 0

    def _plan_milp(self, env, cells, arms, net):
        """
        Точный отбор: одна опция (target, канал) на ячейку, максимум Σ net при лимитах охвата, бюджета,
        10 кампаний и 5000 абонентов на кампанию. Жадный план теряет ячейки, когда групп (seg, target, канал)
        больше 10, и выбирает target ячейки без оглядки на то, с кем её можно объединить в кампанию.
        """
        from scipy.optimize import Bounds, LinearConstraint, milp
        ch = env.channels
        opts = []
        free = [c for c in ch if ch[c]["cost_per_contact"] == 0]
        for k, a in arms.items():
            if k[:2] not in cells or not ("prior" in a.get("src", ("prior",)) or a["n"]):
                continue
            c = cells[k[:2]]
            x = {"cur": k[0], "seg": k[1], "target": k[2], "mu": a["mu"], "sd": math.sqrt(a["var"]), "lcb": _lcb(a),
                 "rank": a["mu"], **c}
            if self._eligible(a, k):
                names = ch
            elif PUSH_FILL and a["mu"] > 0:  # бесплатный канал: риск только в эффекте, ожидание > 0 окупается
                names = free
            else:
                continue
            for name in names:
                v = net(x, name)
                if v > 0:
                    opts.append((v, name, x))
        if not opts:
            return []
        cell_ids = {k: i for i, k in enumerate(dict.fromkeys((o[2]["cur"], o[2]["seg"]) for o in opts))}
        grp_ids = {g: i for i, g in enumerate(dict.fromkeys((o[2]["seg"], o[2]["target"], o[1]) for o in opts))}
        n_o, n_g = len(opts), len(grp_ids)
        nvar = n_o + n_g
        rows, lo, hi = [], [], []

        def row(entries, l, h):
            r = np.zeros(nvar)
            for j, v in entries:
                r[j] += v
            rows.append(r); lo.append(l); hi.append(h)

        for cid in cell_ids.values():
            row([(j, 1) for j, o in enumerate(opts) if cell_ids[(o[2]["cur"], o[2]["seg"])] == cid], 0, 1)
        row([(j, o[2]["n"]) for j, o in enumerate(opts)], 0, env.remaining_contacts)
        row([(j, o[2]["n"] * ch[o[1]]["cost_per_contact"]) for j, o in enumerate(opts)], 0, env.remaining_budget)
        for g, gi in grp_ids.items():
            members = [j for j, o in enumerate(opts) if (o[2]["seg"], o[2]["target"], o[1]) == g]
            row([(j, opts[j][2]["n"]) for j in members] + [(n_o + gi, -5000)], -np.inf, 0)  # кампания ≤ 5000
            for j in members:
                row([(j, 1), (n_o + gi, -1)], -np.inf, 0)  # опция только в открытой группе
        row([(n_o + gi, 1) for gi in grp_ids.values()], 0, 10)
        cost = -np.r_[[o[0] for o in opts], np.zeros(n_g)]
        res = milp(cost, constraints=LinearConstraint(np.array(rows), lo, hi), integrality=np.ones(nvar),
                   bounds=Bounds(0, np.r_[np.ones(n_o), np.full(n_g, 10)]), options={"time_limit": MILP_TIME})
        if res.x is None:
            raise RuntimeError(f"milp: {res.message}")
        chosen = []
        for j, o in enumerate(opts):
            if res.x[j] > 0.5:
                chosen.append({**o[2], "ch": o[1]})
        self.log.append(f"milp: {len(chosen)} ячеек, {sum(res.x[n_o:] > 0.5)} групп, net≈{-res.fun:,.0f}")
        return chosen

    def _plan_greedy(self, env, picks, net, names):
        ch = env.channels
        # охват: сначала самые ценные на контакт
        picks.sort(key=lambda x: x["rank"] * x["S"] / x["n"], reverse=True)
        room, chosen = env.remaining_contacts, []
        for x in picks:
            if x["n"] <= room:
                chosen.append(x)
                room -= x["n"]
        self._chosen = chosen

        # канал: всем push, затем жадный апгрейд по Δnet/Δcost в рамках бюджета
        money = env.remaining_budget
        for x in chosen:
            x["ch"] = names[0]
        while True:
            best = None
            for x in chosen:
                for c in names[names.index(x["ch"]) + 1:]:
                    dcost = (ch[c]["cost_per_contact"] - ch[x["ch"]]["cost_per_contact"]) * x["n"]
                    dnet = net(x, c) - net(x, x["ch"])
                    if dnet > 0 and dnet / max(dcost, 1e-9) >= UPGRADE_MIN_ROI and 0 < dcost <= money and (best is None or dnet / dcost > best[0]):
                        best = (dnet / dcost, x, c, dcost)
            if best is None:
                break
            _, x, c, dcost = best
            x["ch"], money = c, money - dcost
        return chosen

    @staticmethod
    def _pack(chosen, net):
        groups = {}
        for x in chosen:
            groups.setdefault((x["seg"], x["target"], x["ch"]), []).append(x)
        campaigns = []
        for (seg, target, c), xs in groups.items():
            chunk, size = [], 0
            for x in sorted(xs, key=lambda x: x["cur"]) + [None]:
                if x is None or size + x["n"] > 5000:
                    if chunk:
                        v = sum(net(y, c) for y in chunk)
                        campaigns.append((v, v / size, {
                            "campaign_name": f"{seg}_{target}_{c}_{len(campaigns) + 1}",
                            "filter_arpu_segment": seg,
                            "filter_current_tariff": ";".join(y["cur"] for y in chunk),
                            "target_tariff": target, "channel": c}))
                    chunk, size = [], 0
                if x is not None:
                    chunk.append(x)
                    size += x["n"]
        top = sorted(campaigns, key=lambda t: t[0], reverse=True)[:10]
        return [camp for _, _, camp in sorted(top, key=lambda t: t[1], reverse=True)]

    def _expert_report(self, arms):
        """Кто из экспертов предложил пилотируемые и попавшие в план рукава — для логов и ablation."""
        planned = {(x["cur"], x["seg"], x["target"]) for x in self._chosen}
        for name in self.experts:
            mine = {k: a for k, a in arms.items() if name in a.get("src", ())}
            obs = [o for a in mine.values() for o in a.get("obs", [])]
            self.log.append(
                f"expert {name}: arms={len(mine)} pilots={len(obs)} "
                f"mean_obs={np.mean(obs) if obs else float('nan'):.3f} "
                f"hit={np.mean([o > 0 for o in obs]) if obs else float('nan'):.2f} "
                f"planned={len(planned & mine.keys())}"
                + (f" weight={self.weights[name]:.2f}" if name in self.weights else ""))

    def _fallback(self, env, arms):
        tried = [(_lcb(a), k) for k, a in arms.items() if a["n"]]
        if tried and max(tried)[0] > 0:
            cur, seg, target = max(tried)[1]
            return [{"campaign_name": "fallback", "filter_arpu_segment": seg,
                     "filter_current_tariff": cur, "target_tariff": target, "channel": "sms"}]
        # ТЗ требует ≥1 кампании. ponytail: бесплатный push на наименее убыточную ячейку,
        # худший случай — потеря |mu|·0.5·S одной ячейки
        p = env.customer_profile
        S = p.groupby(["current_tariff", "arpu_segment"])["predicted_arpu"].sum()
        pool = {k: a for k, a in arms.items() if a["n"]} or arms
        if pool:
            cur, seg, target = max(pool, key=lambda k: pool[k]["mu"] * S.get(k[:2], 0.0))
        else:  # arms пусты (сбой до гипотез): самая маленькая по ARPU ячейка, любой другой тариф
            cur, seg = S.idxmin()
            target = next(t for t in env.tariffs["tariff_plan_code"] if t != cur)
        return [{"campaign_name": "fallback_push", "filter_arpu_segment": seg,
                 "filter_current_tariff": cur, "target_tariff": target, "channel": "push"}]
