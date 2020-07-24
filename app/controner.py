"""
播放器指标路由
"""
import json
from flask import request, Response, Blueprint
from app.factory import FormatChecker
from app.model import PlayerIndex

player_app = Blueprint('player_app', __name__)
format_handler = FormatChecker()
model_handler = PlayerIndex()


@player_app.route('/index/dot', methods=['POST'])
def get_dot_index():
    input_data = request.data.decode('utf-8')
    if format_handler.fuzz_task_check(input_data):
        input_json = json.loads(input_data)
        result_json = model_handler.get_dot_index(input_json)
        return Response(json.dumps({
            "code": 0,
            "message": "Success",
            "index": {
                "first_frame_time": 111,
                "freeze_time": 60,
                "total_time": 120
                      }
        }), content_type='application/json')
    else:
        return Response(json.dumps({
            "code": -1,
            "message": "input error"}), content_type='application/json')


@player_app.route('/index/cv', methods=['POST'])
def update_cv_data():
    pass


@player_app.route('/index/cv?<task_id>', methods=['GET'])
def get_cv_index():
    pass


@player_app.route('/')
def heart_beat():
    return Response("hello,world")


if __name__ == "__main__":
    pass
