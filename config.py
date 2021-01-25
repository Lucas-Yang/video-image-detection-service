# /usr/bin/env python
# -*- coding: utf-8 -*-
import multiprocessing

# 并行工作进程数
# workers = multiprocessing.cpu_count() * 2 + 1
workers = 4
# 指定每个工作者的线程数
threads = 10
# 监听内网端口
bind = '0.0.0.0:8090'
# 设置守护进程,将进程交给supervisor管理
daemon = 'false'
# 工作模式协程
worker_class = 'gevent'
# 设置最大并发量
worker_connections = 1000
# 设置进程文件目录
pidfile = 'log/gunicorn.pid'
# 设置访问日志和错误信息日志路径
accesslog = 'log/gunicorn_acess.log'
errorlog = 'log/gunicorn_error.log'
# 设置日志记录水平
loglevel = 'DEBUG'
timeout = 120

# celery 任务队列, mongodb 作为backend方便持久化存储
broker_url = 'redis://0.0.0.0:6379/3'
task_track_started = True
result_backend = 'mongodb://burytest:GbnO35lpzAyjkPqSXQTiHwLuDs2r4gcR@172.22.34.102:3301/test' \
                        '?authSource=burytest&replicaSet=bapi&readPreference=primary&appname=MongoDB%2' \
                        '0Compass&ssl=false'
mongodb_backend_settings = {
    "database": "burytest",
    "taskmeta_collection": "video_parsing_tasks_temp"
}
