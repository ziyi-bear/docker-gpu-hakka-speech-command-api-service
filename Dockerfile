FROM nvidia/cuda:10.0-cudnn7-devel

LABEL maintainer="m0724001@gm.nuu.edu.tw"

# 安裝好Ubuntu後需要的動作
RUN apt-get update
# 設定預設語系和編碼為UTF-8
# https://stackoverflow.com/questions/28405902/how-to-set-the-locale-inside-a-debian-ubuntu-docker-container
# https://medium.com/@adsl8212/%E5%A6%82%E4%BD%95%E4%BF%AE%E6%94%B9docker-container%E5%85%A7%E7%9A%84%E8%AA%9E%E7%B3%BB-b3a0cef0b810
RUN apt install locales && \
    sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

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

# 因採用MQTT通訊, 因此不可多線程
CMD [ "python3", "-u", "/app/app.py" ]
