"""
dao层
"""
from app.third_lib.dot_predict import DotVideoIndex
from app.third_lib.model_predict import DeepVideoIndex


class PlayerIndex(object):
    """ 数据获取与数据存储
    """

    def __init__(self):
        pass

    def __write_db(self, info_dict: dict = {}):
        """ 将计算好的指标入库
        :param info_dict:
        :return:
        """
        return

    def __read_db(self, db_cmd):
        """ 查询打点数据
        :param db_cmd:
        :return:
        """
        return

    def get_dot_index(self, video_info: dict):
        """ 获取打点数据, 获取设备上"最新！！"播放行为的打点数据
        :param video_info: {"buvid": xxx, "mid": xxx}
        :return:
        """
        dot_index_handler = DotVideoIndex(video_info)
        dot_video_info_dict = dot_index_handler.get_total_index()
        self.__write_db(dot_video_info_dict)
        return dot_video_info_dict

    def get_cv_index(self, video_info: dict):
        """获取视频计算数据
        :param video_info:
        :return:
        """
        return
