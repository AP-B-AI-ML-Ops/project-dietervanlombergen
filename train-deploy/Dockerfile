FROM python:3.13-slim

RUN pip install -U pip

COPY . /

RUN pip install -r ./requirements.txt --upgrade

RUN prefect config set PREFECT_API_URL=http://orchestration:4200/api

CMD [ "python", "main.py" ]