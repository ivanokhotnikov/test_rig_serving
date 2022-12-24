FROM python:3.10-slim
COPY . /app
RUN pip install --no-cache-dir -r app/configs/requirements.txt
WORKDIR /app
ENTRYPOINT [ "streamlit", "run", "./src/serving.py", "--server.port=8081", "--server.address=0.0.0.0", "--server.maxUploadSize=500", "--server.maxMessageSize=500"]