FROM python:3.10-slim
RUN apt-get update && apt-get install make
COPY . /app
RUN pip install --no-cache-dir -r app/configs/requirements.txt
WORKDIR /app
CMD make serve