FROM python:3.13-slim

RUN pip install -U pip

# We add uvicorn explicitly to ensure MLflow uses the modern stack
RUN pip install psycopg2-binary mlflow uvicorn


CMD ["/bin/bash", "-c", "mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB} --allowed-hosts \"*\"" ]

# --host omdat mlflow anders op 127.0.0.1 runt, en de port niet kan exposen naar buiten
# --port omdat we zeker willen zijn dat mlflow runt op port 5000 (belangrijk voor scripts)
# psycopg2 om mlflow de database te laten accessen
# mlflow ui --host 0.0.0.0 --port 5000 --backend-store-uri postgresql+psycopg2://postgres:postgres@database/mlflow_db