"""
dao层
"""
from app.third_lib.dot_predict import DotVideoIndex
from app.third_lib.cv_predict import DeepVideoIndex, ModelType
from app.third_lib.full_reference_video_quality import VideoSSIM
from app.factory import LogManager


class PlayerIndex(object):
    """ 数据获取与数据存储
    """

    def __init__(self,
                 dot_info_dict: dict = None,
                 cv_info_dict: dict = None,
                 video_quality_dict: dict = None
                 ):
        self.dot_info_dict = dot_info_dict
        self.cv_info_dict = cv_info_dict
        self.video_quality_dict = video_quality_dict
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
        start_app_time = None
        black_frame_list = []
        freeze_frame_list = []
        first_frame_timestamp = None
        deep_index_handler = DeepVideoIndex(self.cv_info_dict)
        for index_type in set(self.cv_info_dict.get("index_types")):
            if index_type == ModelType.FIRSTFRAME.name:
                first_frame_time, cls_results_dict, first_frame_timestamp = deep_index_handler.get_first_frame_time()
            elif index_type == ModelType.BLACKFRAME.name:
                black_frame_list = deep_index_handler.get_black_frame_info()
            elif index_type == ModelType.FREEZEFRAME.name:
                freeze_frame_list = deep_index_handler.get_freeze_frame_info()
            elif index_type == ModelType.BLURREDFRAME.name:
                pass
            elif index_type == ModelType.STARTAPP.name:
                start_app_time, cls_results_dict = deep_index_handler.get_app_start_time()
            else:
                pass

        # 数据筛选，只判断播放阶段的卡顿与黑屏，删除播放前阶段的卡顿与黑屏数据
        if first_frame_time and first_frame_timestamp:
            first_frame_time_step = first_frame_timestamp

            freeze_frame_list = [freeze_frame_dict for freeze_frame_dict in freeze_frame_list
                                 if float(freeze_frame_dict.get("freeze_start_time")) > first_frame_time_step
                                 ]
            # black_frame_list = [black_frame_dict for black_frame_dict in black_frame_list
            #                    if float(black_frame_dict.get("black_start") > first_frame_time_step)
            #                    ]

        cv_index_result = {
            "image_dict": cls_results_dict,
            "start_app_time": start_app_time,
            "first_frame_time": first_frame_time,
            "black_frame_list": black_frame_list,
            "freeze_frame_list": freeze_frame_list
        }
        # self.__write_db()
        return cv_index_result

    def get_video_quality(self):
        """
        :return:
        """
        src_video = self.video_quality_dict.get("src_video")
        target_video = self.video_quality_dict.get("target_video")
        video_quality_handler = VideoSSIM(src_video, target_video)
        ssim_score = video_quality_handler.get_video_ffmpeg_ssim_index()
        return {"ssim_score": ssim_score}


if __name__ == '__main__':
    import json
    cv_info_dict = {"temp_video_path": "/Users/luoyadong/PycharmProjects/PlayerIndex/result/screen.mp4",
                    "index_types": ["FREEZEFRAME", "BLACKFRAME", "FIRSTFRAME"]
                    }
    dot_info_dict = {

    }
    player_handler = PlayerIndex(dot_info_dict=dot_info_dict)
    cv_index_result1 = player_handler.get_dot_index()
    print(json.dumps(cv_index_result1, ensure_ascii=False))
