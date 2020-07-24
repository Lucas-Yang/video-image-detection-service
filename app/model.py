"""
dao层
"""


class PlayerIndex(object):
    """ 数据获取与数据存储
    """

    def __init__(self):
        pass

    def __write_db(self, db_cmd):
        """ 将计算好的指标入库
        :param db_cmd:
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
        """ 获取打点数据
        :param video_info:
        :return:
        """
        return

    def get_cv_index(self, video_info: dict):
        """获取视频计算数据
        :param video_info:
        :return:
        """
        return
