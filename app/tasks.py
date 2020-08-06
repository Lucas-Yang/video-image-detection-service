"""
celery异步任务执行
"""

from celery import Celery
from app.factory import LogManager
from app.model import PlayerIndex
import traceback

logger = LogManager('server.log').logger

brokers = 'redis://127.0.0.1:6379/7'
backend = 'redis://127.0.0.1:6379/8'

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
        model_handler = PlayerIndex(cv_info_dict=cv_info_dict)
        cv_result_index = model_handler.get_cv_index()
        logger.info(cv_result_index)
    except Exception as err:
        logger.error(err)
        logger.error(traceback.format_exc())
        task_success_flag = False
        cv_result_index = {}
    return task_success_flag, cv_result_index


if __name__ == '__main__':
    print('')
