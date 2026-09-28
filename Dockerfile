FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn "fastapi-users[sqlalchemy]" "psycopg[binary]"
COPY . .
# пакет среды не в git: хостинг собирает из клона — тогда берём его из релиза participant-pkg (локальный пакет в приоритете)
ADD https://github.com/andreidevs/CampaignCockpit/releases/download/participant-pkg/participant-pkg.tar.gz /tmp/pkg.tar.gz
RUN if [ ! -f agent_template.py ]; then { [ -L data ] && rm data; } ; tar xzf /tmp/pkg.tar.gz; fi; rm /tmp/pkg.tar.gz
# без пакета сервер падает на старте с ModuleNotFoundError — лучше упасть на сборке и сказать почему
RUN for f in agent_template.py environment.py mock_environment.py scoring_core.py local_eval.py make_submission.py \
        customer_profile.csv data/change_tariff.csv; do \
      test -f "$f" || { echo "нет $f: положите пакет среды в корень репо (README, шаг 1); симлинки наружу в образ не попадают"; exit 1; }; done
COPY --from=web /web/dist web/dist
EXPOSE 8000
# хостинг задаёт порт через $PORT; в docker compose его нет — 8000
CMD ["sh", "-c", "exec uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]
