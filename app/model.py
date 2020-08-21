"""
dao层
"""
from app.third_lib.dot_predict import DotVideoIndex
from app.third_lib.cv_predict import DeepVideoIndex, ModelType
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
        success_flag = True
        dot_video_info_dict = {}
        try:
            dot_video_info_dict = dot_index_handler.get_total_index()
        except Exception as error:
            self.__logger.error(error)
            self.__logger.exception(error)
            success_flag = False
        # self.__write_db(dot_video_info_dict)
        return success_flag, dot_video_info_dict

    def get_cv_index(self):
        """获取视频计算数据
        :return:
        """
        cls_results_dict = {"frame 分类": [[None, None]]}
        first_frame_time = None
        black_frame_list = []
        freeze_frame_list = []

        deep_index_handler = DeepVideoIndex(self.cv_info_dict)
        for index_type in set(self.cv_info_dict.get("index_types")):
            if index_type == ModelType.FIRSTFRAME:
                first_frame_time, cls_results_dict = deep_index_handler.get_first_frame_time()
            elif index_type == ModelType.BLACKFRAME:
                black_frame_list = deep_index_handler.get_black_frame_info()
            elif index_type == ModelType.FREEZEFRAME:
                freeze_frame_list = deep_index_handler.get_freeze_frame_info()
            elif index_type == ModelType.BLURREDFRAME:
                pass
            elif index_type == ModelType.STARTAPP:
                pass
            else:
                pass

        cv_index_result = {
            "image_dict": cls_results_dict,
            "first_frame_time": first_frame_time,
            "black_frame_list": black_frame_list,
            "freeze_frame_list": freeze_frame_list
        }
        # self.__write_db()
        return cv_index_result
