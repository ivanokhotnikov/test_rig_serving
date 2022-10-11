FROM python:3.10-slim
COPY /configs/requirements.txt app/requirements.txt
RUN pip install --no-cache-dir -r app/requirements.txt
COPY /src /app
WORKDIR /app
ENTRYPOINT ["streamlit", "run", "serving.py", "--server.port=8081", "--server.address=0.0.0.0"]