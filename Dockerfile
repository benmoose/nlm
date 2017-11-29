FROM python:3.6

ENV PYTHONUNBUFFERED 1

RUN mkdir /code
WORKDIR /code

COPY . /code/

RUN apt-get update -qq
RUN apt-get install -y libblas-dev liblapack-dev liblapacke-dev gfortran python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
