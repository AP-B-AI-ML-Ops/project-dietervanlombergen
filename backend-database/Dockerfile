FROM postgres:latest

RUN echo "initialising database..."

COPY ./init.sql /docker-entrypoint-initdb.d/

# RUN su - postgres

# RUN echo "SELECT 'CREATE DATABASE mlflow_db' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow_db')\gexec" | psql
