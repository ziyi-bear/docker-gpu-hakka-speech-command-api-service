FROM nvidia/cuda:10.0-cudnn7-devel

LABEL maintainer="m0724001@gm.nuu.edu.tw"

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

RUN apt-get update && \
    apt-get install -y \
    git \
    nano \
    unzip \
    python3 \
    python3-pip \
    locales

RUN pip3 install flask-bootstrap \
    flask-mqtt \
    flask-socketio \
    tensorflow \
    flask

RUN cd /tmp && \
    git clone https://github.com/tensorflow/tensorflow.git && \
    