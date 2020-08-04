"""
播放器指标路由
"""
import json
import os
from datetime import datetime

from flask import request, Response, Blueprint, render_template
from app.factory import FormatChecker, LogManager
from app.model import PlayerIndex
from app.tasks import celery_app, cv_index_task

player_app = Blueprint('player_app', __name__, template_folder='templates')
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
    base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'temp_dir')
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    file_path = os.path.join(base_path, str(datetime.now()) + f.filename)
    f.save(file_path)  # ffmpeg直接读FileStorage的方法暂时没有调研到，所以保存到一个临时文件
    r = cv_index_task.delay({"temp_video_path": file_path})
    task_id = r.task_id
    return Response(json.dumps({
        "code": 0,
        "message": "Success",
        "task_id": task_id
    }), content_type='application/json')



@player_app.route('/index/cv?<task_id>', methods=['GET'])
def get_cv_index():
    return Response("cv index")


@player_app.route('/')
def heart_beat():
    info = {"image_dict": {0: ["", ""], 1: ["", ""], 2: ["", ""]},
            "first_frame_time": 1,
            "stage": ["阶段0: 播放器打开", "阶段1: 播放器加载", "阶段2: 播放器播放", "阶段3: 无关阶段"]
            }
    return render_template('template_reporter.html', info=info)


if __name__ == "__main__":
    pass
