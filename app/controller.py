"""
播放器指标路由
"""
import json
import os
from datetime import datetime

from flask import request, Response, Blueprint, current_app
from app.factory import FormatChecker, LogManager
from app.model import PlayerIndex


player_app = Blueprint('player_app', __name__)
format_handler = FormatChecker()

logger = LogManager("server.log").logger


@player_app.route('/index/dot', methods=['POST'])
def get_dot_index():
    model_handler = PlayerIndex()
    input_data = request.data.decode('utf-8')
    if format_handler.player_index_dot_check(input_data):
        input_json = json.loads(input_data)
        result_json = model_handler.get_dot_index(input_json)
        return Response(json.dumps({
            "code": 0,
            "message": "Success",
            "index": result_json
        }), content_type='application/json')
    else:
        return Response(json.dumps({
            "code": -1,
            "message": "input error"}), content_type='application/json')


@player_app.route('/video/upload', methods=['POST'])
def update_cv_data():
    f = request.files['file']
    logger.info(type(f))
    base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'temp_dir')
    file_path = os.path.join(base_path, str(datetime.now()) + f.filename)
    f.save(file_path)   # ffmpeg直接读FileStorage的方法暂时没有调研到，所以保存到一个临时文件
    model_handler = PlayerIndex(cv_info_dict={"temp_video_path": file_path, "save_path": base_path})
    model_handler.get_cv_index()
    return "1111"


@player_app.route('/index/cv?<task_id>', methods=['GET'])
def get_cv_index():
    return Response("cv index")


@player_app.route('/')
def heart_beat():
    logger.info("testtest")
    return Response("hello,world")


if __name__ == "__main__":
    pass
