#Base image to use
FROM python:3.10.3-slim

#Expose port 8080
EXPOSE 8080

#Copy requirements.txt file into app directory
COPY requirements.txt app/requirements.txt

#Install all requirements in requirements.txt
RUN pip install -r app/requirements.txt

#Copy all files in current directory into app directory
COPY . /app

#Change Working Directory to app directory
WORKDIR /app

#Run the application on port 8080
ENTRYPOINT ["streamlit", "run", "run_detector.py", "--server.port=8080", "--server.address=0.0.0.0"]