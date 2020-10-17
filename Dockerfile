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

RUN pip3 install \
    tensorflow \
    flask

# If STATIC_INDEX is 1, serve / with /static/index.html directly (or the static URL configured)
ENV STATIC_INDEX 1

EXPOSE 80

COPY ./app /app
WORKDIR /app