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
# пакет среды не в git: без него сервер падает на старте с ModuleNotFoundError — лучше упасть на сборке и сказать почему
RUN for f in agent_template.py environment.py mock_environment.py scoring_core.py local_eval.py make_submission.py \
        customer_profile.csv data/change_tariff.csv; do \
      test -f "$f" || { echo "нет $f: положите пакет среды в корень репо (README, шаг 1); симлинки наружу в образ не попадают"; exit 1; }; done
COPY --from=web /web/dist web/dist
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
