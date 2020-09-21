FROM python:3.7
LABEL maintainer="luka luoyadong@bilibili.com"
WORKDIR /my_app

COPY requirements.txt ./
RUN pip install -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt
RUN apt-get update && apt-get -y install libgl1-mesa-glx
COPY . .

EXPOSE 8090
# CMD ["cd", "/my_app"]
CMD ["sh", "/my_app/begin.sh"]