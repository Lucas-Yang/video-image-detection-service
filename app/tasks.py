"""
celery异步任务执行
"""
import os
from celery import Celery
from ffmpy import FFmpeg
from app.factory import LogManager
from app.model import PlayerIndex
import traceback

logger = LogManager('server.log').logger

# plz double check redis database num
#brokers = 'redis://172.22.119.29:6379/3'
#backend = 'redis://172.22.119.29:6379/4'
brokers = 'redis://0.0.0.0:6379/3'
backend = 'redis://0.0.0.0:6379/4'

celery_app = Celery('player-tasks', broker=brokers, backend=backend)


@celery_app.task
def add(x, y):
    return x + y


@celery_app.task
def cv_index_task(cv_info_dict):
    """ 视频计算celery 任务
    :param cv_info_dict: 输入数据
    :return:
    """
    task_success_flag = True
    try:
        file_path = cv_info_dict.get("temp_video_path")
        cfr_video_path = cv_info_dict.get("temp_video_path").split('.mp')[0] + "1.mp4"
        ff = FFmpeg(
            inputs={file_path: None},
            outputs={
                cfr_video_path: '-y -vf fps=32 -vsync cfr'
            }
        )
        ff.run()
        cv_info_dict["temp_video_path"] = cfr_video_path
        model_handler = PlayerIndex(cv_info_dict=cv_info_dict)
        cv_result_index = model_handler.get_cv_index()
        os.remove(cv_info_dict.get("temp_video_path"))  # 删除临时视频文件
        os.remove(file_path)  # 删除临时固定帧率源视频
        logger.info(cv_result_index)
    except Exception as err:
        logger.error(err)
        logger.error(traceback.format_exc())
        task_success_flag = False
        cv_result_index = {}

    return task_success_flag, cv_result_index


if __name__ == '__main__':
    print('')
