# /usr/bin/env python
# -*- coding: utf-8 -*-

"""
播放器指标路由
"""
import json
import os
import time

from flask import request, Response, Blueprint, render_template
from celery.result import AsyncResult
from app.factory import FormatChecker, LogManager
from app.model import PlayerIndex
from app.tasks import celery_app, cv_index_task, video_quality_task

player_app = Blueprint('player_app', __name__, template_folder='templates')
format_handler = FormatChecker()
logger = LogManager("server.log").logger


@player_app.route('/index/dot', methods=['POST'])
def get_dot_index():
    input_data = request.data.decode('utf-8')
    if format_handler.player_index_dot_check(input_data):
        input_json = json.loads(input_data)
        model_handler = PlayerIndex(dot_info_dict=input_json)
        success_flag, result_json = model_handler.get_dot_index()
        if success_flag:
            return Response(json.dumps({
                "code": 0,
                "message": "Success",
                "index": result_json
            }), content_type='application/json')
        else:
            return Response(json.dumps({
                "code": -2,
                "message": "calculate index error, plz check log",
            }), content_type='application/json')
    else:
        return Response(json.dumps({
            "code": -1,
            "message": "input error"}), content_type='application/json')


@player_app.route('/video/upload', methods=['POST'])
def update_cv_data():
    if format_handler.player_index_cv_check(request):
        f = request.files['file']
        base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'temp_dir')
        if not os.path.exists(base_path):
            os.mkdir(base_path)
        file_path = os.path.join(base_path, str(time.time()) + f.filename)
        # ffmpeg直接读FileStorage的方法暂时没有调研到，所以保存到一个临时文件
        f.save(file_path)
        logger.info(file_path)
        r = cv_index_task.delay({"temp_video_path": file_path, "index_types": request.form.getlist("index_types")})
        task_id = r.task_id
        return Response(json.dumps({
            "code": 0,
            "message": "Success",
            "task_id": task_id
        }), content_type='application/json')
    else:
        return Response(json.dumps({
            "code": -1,
            "message": "input error"}), content_type='application/json')


@player_app.route('/index/cv', methods=['GET'])
def get_cv_index():
    task_id = request.args.get('task_id')
    model_handler = PlayerIndex()
    if task_id:
        res = AsyncResult(task_id, app=celery_app)
        if res.failed():
            return Response(json.dumps({
                "code": -2,
                "message": "task failed, plz try again"}), content_type='application/json')
        elif res.status == "PENDING" or res.status == "RETRY" or res.status == "STARTED":
            history_result = model_handler.get_history_parsing_task(task_id)
            if history_result:
                return Response(json.dumps({
                    "code": 0,
                    "data": history_result,
                    "message": "Success"}), content_type='application/json')
            else:
                return Response(json.dumps({
                    "code": -4,
                    "message": "plz wait a moment"}), content_type='application/json')

        elif res.status == "SUCCESS":
            if res.result[0]:
                logger.info(res)
                model_handler.save_tasks_db({"task_id": task_id,
                                             "task_result": res.result[1]}
                                            )
                return Response(json.dumps({
                    "code": 0,
                    "data": res.result[1],
                    "message": "Success"}), content_type='application/json')
                # return render_template('template_reporter.html', info=res.result[1])
            else:
                return Response(json.dumps({
                    "code": -3,
                    "message": "inner error"}), content_type='application/json')
        else:
            return Response(json.dumps({
                "code": -3,
                "message": "inner error"}), content_type='application/json')
    else:
        return Response(json.dumps({
            "code": -1,
            "message": "input error, make sure task id is correct"}), content_type='application/json')


@player_app.route('/')
def heart_beat():
    info = {"image_dict": {0: ["", ""], 1: ["", ""], 2: ["", ""]},
            "first_frame_time": 1,
            "stage": ["阶段0: 播放器打开", "阶段1: 播放器加载", "阶段2: 播放器播放", "阶段3: 无关阶段"]
            }
    return render_template('template_reporter.html', info=info)


@player_app.route('video/ssim', methods=['POST'])
def get_ssim_index():
    if format_handler.ssim_index_checker(request):
        f_src = request.files['file_src']
        f_target = request.files['file_target']
        base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'temp_dir')
        if not os.path.exists(base_path):
            os.mkdir(base_path)
        file_src_path = os.path.join(base_path, str(time.time()) + f_src.filename)
        file_target_path = os.path.join(base_path, str(time.time()) + f_target.filename)
        f_src.save(file_src_path)
        f_target.save(file_target_path)
        model_handler = PlayerIndex(video_quality_dict={"src_video": file_src_path, "target_video": file_target_path})
        video_quality_result_index = model_handler.get_video_quality()
        os.remove(file_src_path)  # 删除临时视频文件
        os.remove(file_target_path)  # 删除临时固定帧率源
        return Response(json.dumps({
            "code": 0,
            "message": "Success",
            "data": video_quality_result_index
        }), content_type='application/json'
        )
    else:
        return Response(json.dumps({
            "code": -1,
            "message": "input error"}), content_type='application/json')