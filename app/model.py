"""
dao层
"""
from app.third_lib.dot_predict import DotVideoIndex
from app.third_lib.cv_predict import DeepVideoIndex
from app.factory import LogManager


class PlayerIndex(object):
    """ 数据获取与数据存储
    """

    def __init__(self, dot_info_dict: dict = None, cv_info_dict: dict = None):
        self.dot_info_dict = dot_info_dict
        self.cv_info_dict = cv_info_dict
        self.__logger = LogManager("server.log").logger

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

    def get_dot_index(self):
        """ 获取打点数据, 获取设备上"最新！！"播放行为的打点数据
        :param : {"buvid": xxx, "mid": xxx}
        :return:
        """
        dot_index_handler = DotVideoIndex(self.dot_info_dict)
        dot_video_info_dict = dot_index_handler.get_total_index()
        self.__write_db(dot_video_info_dict)
        return dot_video_info_dict

    def get_cv_index(self):
        """获取视频计算数据
        :return:
        """
        deep_index_handler = DeepVideoIndex(self.cv_info_dict)
        first_frame_time, cls_results_dict = deep_index_handler.get_first_video_time()
        cv_index_result = {
            "image_dict": cls_results_dict,
            "first_frame_time": first_frame_time,
            "stage": ["阶段0: 播放器打开", "阶段1: 播放器加载", "阶段2: 播放器播放", "阶段3: 无关阶段"]
        }
        # self.__write_db()
        return cv_index_result
