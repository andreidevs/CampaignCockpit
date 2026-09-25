# Настройки

## Переменные окружения
Все необязательные. Локально их можно положить в `.env`: его читают `docker compose` и `server.py`.

| Переменная | Где | По умолчанию | Назначение |
|---|---|---|---|
| `OPENAI_API_KEY` | агент | — | ключ LLM-эксперта |
| `OPENAI_MODEL` | агент | `gpt-4o-mini` | модель OpenAI |
| `OPENAI_BASE_URL` | агент | `https://api.openai.com/v1` | OpenAI-совместимый endpoint |
| `OPENROUTER_API_KEY` | агент | — | если задан, LLM идёт через OpenRouter (приоритет над OpenAI) |
| `LLM_MODEL` | агент | `openai/gpt-4o-mini` | модель OpenRouter |
| `DATABASE_URL` | UI | `postgresql://cockpit:cockpit@localhost:5432/cockpit` | Postgres; `sqlite:///путь.db` — SQLite (так работает [десктоп](desktop.md)) |
| `DB_SCHEMA` | UI | `public` | схема БД (разводит стенды) |
| `AUTH_USERS` | UI | — | `email:пароль:роль,...`, заводятся при пустой таблице |
| `AUTH_DEMO` | UI | `1` в docker compose | `1` — без `AUTH_USERS` завести демо-пользователей; иначе сервер не стартует на пустой таблице |
| `AUTH_SECRET` | UI | `dev-secret-change-me` | секрет сессий; на стенде задайте свой |
| `UPLOAD_DIR` | UI | `uploads/` | куда кладутся загруженные выгрузки (в docker — volume) |
| `ENV_FILE` | UI | `.env` в корне репо | откуда `server.py` читает `.env` (десктоп — каталог данных приложения) |
| `LAB_QUICK` | лаборатория | — | `1` — быстрая матрица тестов (для self-check) |

## Константы агента
Флаги отброшенных экспериментов (`LCB_K_HIGH`, `PILOT_POS_ONLY`, `B2`, `PUSH_K`, `FLAT_PRIOR`, `SCALE_EXPLORE`, `SEED_BIG`, `TAIL_*`, `TARGET_SD`, `SEG_SD`, `CORR_KG`) удалены из кода; их замеры — в [experiments.md](experiments.md), код — в коммите `3554c1c` ветки `post-deadline`.

Верх `agent.py`. Значения по умолчанию — поведение, на котором собран `submission.csv`. Часть констант может менять версия [лаборатории](lab.md) (allowlist `PATCH_TARGETS` в `lab.py` с границами); promote переписывает их в `agent.py`.

### Модель и отбор
| Константа | По умолчанию | Смысл |
|---|---|---|
| `NOISE_STD` | 0.804 | шум на абонента из документации среды; пилот на n весит `0.804/√n` |
| `PRIOR_STD` | 0.25 | ширина prior: насколько не доверяем истории |
| `ARMS_PER_CELL` | 4 | гипотез prior на ячейку |
| `ARMS_BIG_FRAC` | 0.10 | ячейка с долей ARPU ≥ порога получает все переходы из истории (сейчас 3 крупнейшие HIGH-ячейки) |
| `LCB_K` | 0.5 | в план идёт `μ − k·σ > 0` |
| `LCB_K_PILOT` | 0.5 | k для рукавов после пилота |
| `PRIOR_SD` | `"share"` | ширина prior: `"share"` — √(SD0² + (SD_SHARE·доля перехода)²), `"flat"` — `PRIOR_STD` |
| `SD0`, `SD_SHARE` | 0.03, 1.0 | постоянная часть и часть от доли перехода в ширине prior |
| `JOINT` | `True` | совместный апостериор θ = β0 + β1·μ_истории + u |
| `B0`, `B1` | 0.03, 0.5 | sd общего сдвига и наклона доверия к истории |
| `SCALE_AT` | `"end"` | когда подбирать масштаб prior по правдоподобию пилотов: `"end"` — перед планом, `"each"` — после каждого пилота |
| `SCALE_MIN_OBS`, `SCALE_GRID_A`, `SCALE_GRID_B` | 3, (0.5…8), (0.25…2) | минимум пилотов для подбора и сетка масштабов |
| `SCALE_MODE` | `"bma"` | `"bma"` — усреднение апостериора по всей сетке масштабов с весами правдоподобия; `"ml"` — одна лучшая точка |
| `PLANNER` | `"milp"` | `"milp"` — точный отбор под все лимиты (scipy), `"greedy"` — прежний жадный |
| `MILP_TIME` | 20 | секунд на решатель |
| `PUSH_FILL` | `True` | рукав с μ > 0, но LCB ≤ 0 может идти в план бесплатным push |
| `PILOT_VALUE` | `"voi"` | выбор пилота: `"voi"` — knowledge gradient минус цена пилота с учётом его собственной выгоды, `"ei"` — expected improvement |
| `ADAPT`, `ADAPT_AFTER`, `ADAPT_MODE` | `True`, 8, `"exploit"` | после 8 пилотов: если масштаб prior по правдоподобию на минимуме сетки (история надёжна), оставшиеся пилоты — полного размера на лучших кандидатах плана; `"stop"` — разведка заканчивается |
| `EI_STOP` | 0.005 | стоп разведки, когда EI < 0.5% стартового |
| `USE_FIT` | `True` | поправка prior по трафику (см. [experiments.md](experiments.md)) |

### Пилоты
| Константа | По умолчанию | Смысл |
|---|---|---|
| `PILOT_CH` | `sms` | канал пилотов: в 1.69 раза информативнее push на контакт |
| `PILOT_FRAC`, `PILOT_MIN`, `PILOT_MAX` | 0.08, 60, 200 | размер пилота — доля ячейки в пределах |
| `TIME_LIMIT` | 480 | дедлайн `act`, с (ТЗ даёт 10 минут) |

### LLM и privacy
| Константа | По умолчанию | Смысл |
|---|---|---|
| `USE_LLM` | `True` | включить LLM-эксперта (нужен ключ) |
| `LLM_PER_CELL` | 2 | гипотез LLM на ячейку |
| `LLM_CLIP` | 0.5 | потолок \|оценки\| LLM |
| `LLM_GATE` | 1.0 | LLM-рукав только в ячейках, где лучший prior μ < порога |
| `LLM_PROMPT_EXTRA` | `""` | дописывается к инструкции LLM (patch target лаборатории) |
| `PRIVACY_MODE` | `aggregate_only` | `synthetic_only` — зашумлённые медианы и n; `debug_safe` — ни одного числа об абонентах |
| `K_MIN` | 10 | ячейки меньше — без статистик |
| `LLM_FIELDS` | `ARPU_3m_avg`, `DATA_VOLUME`, `OUT_LOC_OFFNET_MIN` | allowlist полей для LLM |

### CatBoost-prior (выключен)
Подробно — [catboost.md](catboost.md).

| Константа | По умолчанию | Смысл |
|---|---|---|
| `PRIOR_MODEL` | `hist` | `catboost_shift` / `catboost_full` |
| `PRIOR_ALPHA_MAX`, `PRIOR_ALPHA_K` | 0.7, 30 | full: `α(n) = ALPHA_MAX·n/(n+K)` |
| `RISK_LAMBDA` | 0.0 | штраф `μ −= λ·u` |
| `PRIOR_WEIGHT` | `rows` | `arpu` — среднее с весом `predicted_arpu` |
| `SUPPORT_MIN` | 10 | меньше переходов — модель может только понизить μ |

### Эвристики (выключены, включает только версия, прошедшая gate)
| Константа | По умолчанию | Альтернатива |
|---|---|---|
| `PILOT_SIZING` | `adaptive` | большой пилот только спорным рукавам, уверенные не перепроверяем; `fixed`: всем `PILOT_FRAC` ячейки |
| `PILOT_BUDGET_SHARE`, `PILOT_CONTACT_SHARE` | 1.0 | доля бюджета / охвата на пилоты |
| `RANK_BY` | `mu` | `lcb`: target и очередь охвата по `μ − LCB_K·σ` |
| `CHANNEL_MU` | `mu` | `lcb`: апгрейд канала по консервативной оценке |
| `UPGRADE_MIN_ROI` | 0.0 | апгрейд канала только при Δnet/Δcost ≥ порога |
| `META_CONTROLLER` | `False` | вес LLM = hit rate его пилотов |
