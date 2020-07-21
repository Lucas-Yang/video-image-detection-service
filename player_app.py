# /usr/bin/env python
# -*- coding: utf-8 -*-

import json
from flask import Flask, request, Response
from celery.result import AsyncResult
from app.factory import LogManager


logger = LogManager('server.log').logger
app = Flask(__name__)


@app.route('/v1/index/dot', methods=['POST'])
def get_dot_index():
    pass


@app.route('/v1/index/cv', methods=['POST'])
def update_cv_data():
    pass


@app.route('/v1/index/cv?<task_id>', methods=['GET'])
def get_cv_index():
    pass


@app.route('/')
def heart_beat():
    return Response("hello,world")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=2233, debug=False)
