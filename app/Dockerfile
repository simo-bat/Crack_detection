FROM python:3.8
COPY . /usr/app
EXPOSE 5000
WORKDIR /usr/app
RUN pip install -r requirements.txt
CMD ["python", "my_api.py", "--host=0.0.0.0"] 