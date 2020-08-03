"""
播放器指标路由
"""
import json
import os
from flask import request, Response, Blueprint
from app.factory import FormatChecker, LogManager
from app.model import PlayerIndex

player_app = Blueprint('player_app', __name__)
format_handler = FormatChecker()
model_handler = PlayerIndex()
logger = LogManager("server.log").logger


@player_app.route('/index/dot', methods=['POST'])
def get_dot_index():
    input_data = request.data.decode('utf-8')
    if format_handler.fuzz_task_check(input_data):
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
    logger.info("111111")
    f = request.files['file']
    # basepath = os.path.dirname(os.path.realpath(__file__))
    base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'temp_dir')
    file_path = os.path.join(base_path, f.filename)
    f.save(file_path)
    model_handler.get_cv_index({"video_path": file_path, "save_path": base_path})
    return "1111"


@player_app.route('/index/cv?<task_id>', methods=['GET'])
def get_cv_index():
    return Response("cv index")


@player_app.route('/')
def heart_beat():
    return Response("hello,world")


if __name__ == "__main__":
    pass
