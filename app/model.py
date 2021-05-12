"""
dao层
"""
from app.factory import LogManager, MyMongoClient
from app.third_lib.cv_predict import DeepVideoIndex, ModelType, VideoSilenceDetector, VideoColourCastDetector
from app.third_lib.dot_predict import DotVideoIndex
from app.third_lib.full_reference_video_quality import VideoSSIM, VideoVMAF
from app.third_lib.image_quality import ImageQualityIndexGenerator


class PlayerIndex(object):
    """ 视频质量数据获取与数据存储
    """

    def __init__(self,
                 dot_info_dict: dict = None,
                 cv_info_dict: dict = None,
                 video_quality_dict: dict = None,
                 silence_info_dict: dict = None,
                 colour_cast_dict: dict = None
                 ):
        self.dot_info_dict = dot_info_dict
        self.cv_info_dict = cv_info_dict
        self.video_quality_dict = video_quality_dict
        self.silence_info_dict = silence_info_dict
        self.colour_cast_dict = colour_cast_dict
        self.__logger = LogManager("server.log").logger
        self.__db_handler = MyMongoClient()

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

    def get_history_parsing_task(self, task_id):
        """ 查询持久化的任务id，防止任务队列重启后任务丢失
        :param task_id:
        :return:
        """
        result = self.__db_handler.query("video_parsing_tasks", {"task_id": task_id}).sort("create_time", -1).limit(1)
        for data in result:
            return data.get("task_result", None)
        return

    def save_tasks_db(self, result_dict):
        """ 同步任务执行结果，理论上起一个定时任务线程，定时同步redis消息队列数据到mongodb储存持久化更合理
        :param result_dict:
        :return:
        """
        try:
            self.__db_handler.insert("video_parsing_tasks", result_dict)
        except Exception as err:
            self.__logger.error(err)
            raise Exception(err)

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
            elif index_type in (ModelType.STARTAPP.name,
                                ModelType.STARTAPPIQIYI.name,
                                ModelType.STARTAPPIXIGUA.name,
                                ModelType.STARTAPPTENCENT.name,
                                ModelType.STARTAPPYOUKU.name,
                                ModelType.STARTAPPDOUYIN.name,
                                ModelType.STARTAPPCOMIC.name
                                ):
                start_app_time, cls_results_dict = deep_index_handler.get_app_start_time(index_type)
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

    def get_video_quality_vmaf(self):
        """获取视频vmaf主观质量评分
        """
        input_video = self.video_quality_dict.get("input_video")
        refer_video = self.video_quality_dict.get("refer_video")
        video_quality_handler = VideoVMAF(input_video, refer_video)
        vmaf_score = video_quality_handler.get_video_vmaf_score()
        return {"vmaf_score": vmaf_score}

    def get_silence_index(self):
        """获取音视频静音时间戳
        """
        video_path = self.silence_info_dict.get("video_path")
        # start_time = self.silence_info_dict.get("start_time")
        # end_time = self.silence_info_dict.get("end_time")
        silence_index_handler = VideoSilenceDetector(video_path=video_path, )
        silence_timestamps = silence_index_handler.get_silent_times()
        return {"silence_timestamps": silence_timestamps}

    def get_colour_cast_index(self):
        """无参考偏色检测"""
        video_path = self.colour_cast_dict.get("video_path")
        colour_cast_handler = VideoColourCastDetector(video_path=video_path)
        colour_space_info = colour_cast_handler.get_space_info()
        return colour_space_info

    def get_colour_cast_index_with_reference(self):
        """有参考偏色检测"""
        src_video_path = self.colour_cast_dict.get("src_video_path")
        target_video_path = self.colour_cast_dict.get("target_video_path")
        colour_cast_handler = VideoColourCastDetector(src_video_path=src_video_path,
                                                      target_video_path=target_video_path)
        average_chroma = colour_cast_handler.get_average_chroma()
        return average_chroma


class ImageIndex(object):
    """ 图像质量数据获取与存储
    """

    def __init__(self, quality_file):
        self.quality_file = quality_file
        self.image_index_handler = ImageQualityIndexGenerator(self.quality_file)
        self.__logger = LogManager("server.log").logger

    def black_white_frame_detection(self):
        """黑白屏检测
        :return:
        """
        gaussian_score = self.image_index_handler.get_black_white_frame_score()
        if gaussian_score < 10:
            return True
        else:
            return False

    def blurred_frame_detection(self):
        """花屏检测
        0: 马赛克
        1: 扭曲
        2: 正常
        :return:
        """
        predict_result = self.image_index_handler.get_if_blurred_frame()
        if predict_result == -1:
            return predict_result
        elif predict_result == 2:
            return False
        else:
            return True

    def frame_ocr(self):
        """ 图像ocr
        :return:
        """
        ocr_result_list = self.image_index_handler.get_ocr_result_list()
        self.__logger.info("ocr:{}".format(ocr_result_list))
        return ocr_result_list

    def frame_horizontal_portrait_detect(self):
        """ 视频帧横竖屏判断(是否是拼接视频)
        :return:
        """
        detect_res = self.image_index_handler.get_horizontal_portrait_frame_size()
        self.__logger.info("horizontal_portrait_detect: {}".format(detect_res))
        return detect_res

    def frame_clarity_detect(self):
        """视频帧清晰度检测（使用NRSS算法）
        :return:
        """
        detect_res = self.image_index_handler.get_image_clarity()
        self.__logger.info("frame_clarity_detect: {}".format(detect_res))
        return detect_res

    def green_frame_detect(self):
        """视频帧绿屏检测
        :return:
        """
        detect_res = self.image_index_handler.get_green_image()
        self.__logger.info("green_frame_detect: {}".format(detect_res))
        return detect_res

    def frame_colorlayer_detect(self):
        """ 颜色区域检测，分为红 绿 蓝
        :return:
        """
        detect_res = self.image_index_handler.get_image_colorlayer()
        self.__logger.info("color_dict: {}".format(detect_res))
        return detect_res


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
