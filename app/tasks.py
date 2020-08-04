from celery import Celery
from app.factory import LogManager
from app.model import PlayerIndex

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
    except Exception as err:
        logger.error(err)
        task_success_flag = False
    return task_success_flag, cv_result_index
