"""
celery异步任务执行
"""
import json
import os
import traceback

from celery import Celery
from ffmpy import FFmpeg

from app.data import VideoQualityItem
from app.factory import LogManager
from app.model import PlayerIndex

logger = LogManager('server.log').logger

celery_app = Celery('player-tasks')
celery_app.config_from_object('config')


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
        if (
                VideoQualityItem.STARTAPPDOUYIN.name in cv_info_dict.get("index_types")
                or VideoQualityItem.STARTAPP.name in cv_info_dict.get("index_types")
                or VideoQualityItem.STARTAPPCOMIC.name in cv_info_dict.get("index_types")
                or VideoQualityItem.STARTAPPIXIGUA.name in cv_info_dict.get("index_types")
                or VideoQualityItem.STARTAPPTENCENT.name in cv_info_dict.get("index_types")
                or VideoQualityItem.STARTAPPYOUKU.name in cv_info_dict.get("index_types")
                or VideoQualityItem.STARTAPPIQIYI.name in cv_info_dict.get("index_types")
        ):
            cfr_video_path = cv_info_dict.get("temp_video_path").split('.mp')[0] + "1.mp4"
            logger.info(cfr_video_path)
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
            cv_result_index = json.loads(json.dumps(cv_result_index))
            os.remove(cv_info_dict.get("temp_video_path"))  # 删除临时视频文件
            os.remove(file_path)  # 删除临时固定帧率源视频
            logger.info(cv_result_index)
        else:
            cv_info_dict["temp_video_path"] = file_path
            model_handler = PlayerIndex(cv_info_dict=cv_info_dict)
            logger.info(cv_info_dict)
            cv_result_index = model_handler.get_cv_index()
            cv_result_index = json.loads(json.dumps(cv_result_index))
            os.remove(cv_info_dict.get("temp_video_path"))  # 删除临时视频文件
            logger.info(cv_result_index)

    except Exception as err:
        logger.error(err)
        logger.error(traceback.format_exc())
        task_success_flag = False
        cv_result_index = {}
        os.remove(cv_info_dict.get("temp_video_path"))  # 删除临时视频文件

    return task_success_flag, cv_result_index


@celery_app.task
def video_quality_task(video_quality_dict):
    """ 有源参考视频检测任务
    :param video_quality_dict:
    :return:
    """
    task_success_flag = True
    try:
        print(video_quality_dict)
        model_handler = PlayerIndex(video_quality_dict=video_quality_dict)
        video_quality_result_index = model_handler.get_video_quality()
        video_quality_result_index = json.loads(json.dumps(video_quality_result_index))
        os.remove(video_quality_dict.get("src_video"))  # 删除临时视频文件
        os.remove(video_quality_dict.get("target_video"))  # 删除临时固定帧率源视频
        logger.info(video_quality_result_index)
    except Exception as err:
        logger.error(err)
        logger.error(traceback.format_exc())
        task_success_flag = False
        video_quality_result_index = {}
        os.remove(video_quality_dict.get("src_video"))  # 删除临时视频文件
        os.remove(video_quality_dict.get("target_video"))  # 删除临时固定帧率源视频
    return task_success_flag, video_quality_result_index


if __name__ == '__main__':
    print(VideoQualityItem.STARTAPPDOUYIN)
