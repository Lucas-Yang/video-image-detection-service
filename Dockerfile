FROM cntrump/ubuntu-ffmpeg:4.2.2
LABEL maintainer="luka luoyadong@bilibili.com"
COPY requirements.txt ./

RUN apt-get update && apt-get install -y \
    --no-install-recommends apt-utils \
    libgl1-mesa-glx
RUN apt-get install -y libsm6 libxext6
RUN apt-get install -y libxrender-dev

RUN apt-get install -y  python3.6 python3-pip
RUN apt-get install -y redis-server
RUN pip3 install -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt
RUN pip3 install --upgrade https://github.com/celery/celery/tarball/master

RUN apt-get install git -y
RUN git clone https://github.com/Netflix/vmaf.git

RUN apt-get install nasm
RUN apt-get install ninja-build meson

WORKDIR /vmaf/libvmaf
RUN meson build --buildtype release
RUN ninja -vC build
RUN ninja -vC build install

RUN useradd -u 8877 work --create-home --no-log-init --shell /bin/bash

WORKDIR /home/work/my_app
RUN chown -R work:work /home/work/my_app


USER work
COPY --chown=work:work . .
RUN ./ffmpeg_configure --enable-libvmaf
COPY --chown=work:work model /home/work
COPY --chown=work:work model /home/work

RUN export C_FORCE_ROOT="true"
RUN export COLUMNS=80
EXPOSE 8090

CMD ["sh", "/home/work/my_app/begin.sh"]