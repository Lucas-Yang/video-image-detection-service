FROM cntrump/ubuntu-ffmpeg:4.2.2
LABEL maintainer="luka luoyadong@bilibili.com"
WORKDIR /my_app

COPY requirements.txt ./

RUN apt-get update && apt-get install -y \
    --no-install-recommends apt-utils \
    libgl1-mesa-glx
RUN apt-get install -y  python3.6 python3-pip

RUN pip3 install -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt
RUN pip3 install --upgrade https://github.com/celery/celery/tarball/master
RUN export C_FORCE_ROOT="true"
RUN export COLUMNS=80
COPY . .
EXPOSE 8090

CMD ["sh", "/my_app/begin.sh"]