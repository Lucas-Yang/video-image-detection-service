FROM python:3.7
LABEL maintainer="luka luoyadong@bilibili.com"
COPY requirements.txt ./
RUN pip install -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt
COPY app ./
COPY log ./
COPY run.py ./
COPY start.sh ./
COPY config.py ./

WORKDIR /app
EXPOSE 8090
CMD ["gunicorn", "-c", "config.py", "run:app"]