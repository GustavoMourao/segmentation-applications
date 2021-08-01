FROM python:3.6-slim-stretch

ADD requirements.txt /
RUN pip install -r /requirements.txt

ADD . /app_tags
WORKDIR /app_tags

EXPOSE 5002
CMD [ "python" , "app_tags.py"]