"""
模型预测封装类
通过调用该类获取视频播放质量指标
"""
import json
import queue
import re
import subprocess
import sys
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from functools import wraps

import cv2
import ffmpeg
import numpy as np
import requests
from PIL import Image

from app.factory import LogManager, MyThread

thread_executor = ThreadPoolExecutor(max_workers=10)



def my_async_decorator(f):
    """ 基于ThreadPoolExecutor的多线程装饰器, 返回future对象，通过调用task.result()获取执行结果
    :param f:
    :return:
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        task = thread_executor.submit(f, *args)
        return task

    return wrapper


def my_async(f):
    """DeepVideoIndex类函数的异步装饰器, 返回执行器，执行完后的结果通过调用task.get_result()
    :param f:
    :return:
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        task = MyThread(f, args=args, kwargs=kwargs)
        task.start()
        return task

    return wrapper


class ModelType(Enum):
    """模型调用枚举，目前暂定支持五种模型，分别调用不同的模型服务
    """
    FIRSTFRAME = 1  # 播放首帧
    STARTAPP = 2  # app启动时间
    FREEZEFRAME = 3  # 卡顿
    BLURREDFRAME = 4  # 花屏
    BLACKFRAME = 5  # 黑屏
    STARTAPPTENCENT = 6  # app启动时间-腾讯
    STARTAPPIQIYI = 7  # app启动时间-爱奇艺
    STARTAPPYOUKU = 8  # app启动时间-优酷
    STARTAPPIXIGUA = 9  # app启动时间-西瓜视频
    STARTAPPDOUYIN = 10  # app启动时间-抖音
    STARTAPPCOMIC = 11


class FirstFrameTimer(object):
    """首帧获取类
    """

    def __init__(self,
                 frame_info_dict: dict = None,
                 stage_name_list: list = None
                 ):
        """
        :param frame_info_dict: 分帧预测信息字典{"frame_url": ["time_step", "predict_result"]}
        :param stage_name_list: 各个阶段的别名，用于生成报告
        """

        if stage_name_list is None:
            stage_name_list = ["阶段0: 播放器打开", "阶段1: 播放器加载", "阶段2: 播放器播放", "阶段3: 无关阶段"]
        self.frame_info_dict = frame_info_dict  # frame时间戳映射表
        self.first_frame_time = 0
        self.stage_name_list = stage_name_list
        self._cls_dict = {}  # 整理后的字典 {"frame 分类": [["img_url1", "time_step1"], ["img_url2", "time_step2"]]}

    def _sort_reformat_result(self):
        """ 排序整理好后的分类dict
        :return:
        """
        for key, value in self._cls_dict.items():
            value.sort(key=lambda x: x[1])

    def _reformat_result(self):
        """ 将整个视频图片的预测结构分类整理成一个dict
        :return:
        """
        for img_url, time_result_list in self.frame_info_dict.items():
            if int(time_result_list[1]) in self._cls_dict:
                self._cls_dict[int(time_result_list[1])].append([img_url, float(time_result_list[0])])
            else:
                self._cls_dict[int(time_result_list[1])] = [[img_url, float(time_result_list[0])]]

    def get_first_frame_time(self):
        """
        :return:
        """

        self._reformat_result()
        self._sort_reformat_result()
        start_frame_timestamp = 0  # 开始播放帧时间戳
        # 正常逻辑，每个阶段都正常识别
        if len(self._cls_dict.get(0, [])) and len(self._cls_dict.get(2, [])):
            start_frame_index = 0
            while self.first_frame_time <= 0:
                self.first_frame_time = float(self._cls_dict.get(2)[start_frame_index][1]) - \
                                        float(self._cls_dict.get(0)[0][1])
                start_frame_index += 1
            start_frame_timestamp = self._cls_dict.get(2)[start_frame_index][1]

        # 当播放完成阶段没有时候，返回-1,给上层判断
        elif len(self._cls_dict.get(1, [])) and len(self._cls_dict.get(2, [])):
            self.first_frame_time = float(self.frame_info_dict.get(self._cls_dict.get(2)[0][0])[0]) - \
                                    float(self.frame_info_dict.get(self._cls_dict.get(1)[0][0])[0])
        else:
            self.first_frame_time = -1

        ordered_dict = OrderedDict(sorted(self._cls_dict.items(), key=lambda obj: obj[0]))
        return self.first_frame_time, ordered_dict, start_frame_timestamp


class StartAppTimer(FirstFrameTimer):
    """ 启动时间计算类
    """

    def __init__(self, start_app_dict=None):
        self.stage_name_list = ["阶段0：app打开", "阶段1：app推荐页加载", "阶段2：app正确启动页面", "阶段3：其他无关页面"]
        super(StartAppTimer, self).__init__(start_app_dict)
        # super.__init__(start_app_dict)


class PlayerFreezeScreenWatcher(object):
    """ 卡顿计算类
    """

    def __init__(self, video_info: dict = None):
        self.video_info_dict = video_info
        self.__logger = LogManager("server.log").logger

    def get_freeze(self):
        """
        :return:
        """
        freeze_result_list = []

        temp_video_path = self.video_info_dict.get("temp_video_path", None)
        if temp_video_path:
            stream = ffmpeg.input(temp_video_path)
            stream = ffmpeg.filter_(stream, 'freezedetect', n=0.0001, d=0.3)
            stream = ffmpeg.output(stream, 'pipe:', format='null')
            out, err = stream.run(quiet=True, capture_stdout=True)
            freeze_start_list = []
            freeze_duration_list = []
            freeze_end_list = []
            for line_info in err.decode().strip().split('\n'):
                if "[freezedetect" in line_info:
                    try:
                        if "freeze_start" in line_info:
                            match_objects_start = re.match(".*freeze_start: (.*)", line_info, re.M | re.I)
                            freeze_start_list.append(match_objects_start.group(1))
                        elif "freeze_duration" in line_info:
                            match_objects_duration = re.match(".*freeze_duration: (.*)", line_info, re.M | re.I)
                            freeze_duration_list.append(match_objects_duration.group(1))
                        elif "freeze_end" in line_info:
                            match_objects_duration = re.match(".*freeze_end: (.*)", line_info, re.M | re.I)
                            freeze_end_list.append(match_objects_duration.group(1))
                        else:
                            continue
                    except Exception as error:
                        self.__logger.error(error)
                        continue
            for freeze_start, freeze_duration, freeze_end in zip(freeze_start_list, freeze_duration_list,
                                                                 freeze_end_list):
                freeze_result = {"freeze_start_time": freeze_start, "freeze_duration_time": freeze_duration,
                                 "freeze_end_time": freeze_end}
                freeze_result_list.append(freeze_result)
            if len(freeze_start_list) > len(freeze_duration_list):
                freeze_result_list.append({"freeze_start_time": freeze_start_list[-1],
                                           "freeze_duration_time": "till-end",
                                           "freeze_end_time": "end"
                                           })
        return freeze_result_list


class PlayerBlurredScreenWatcher(object):
    """ 花屏计算类
    """

    def __init__(self):
        pass

    def get_blurred_screen(self, predict_result_list: list):
        """
        :return:
        """
        return None, None


class PlayerBlackScreenWatcher(object):
    """ 黑屏计算类
    """

    def __init__(self, video_info: dict = None):
        self.video_info_dict = video_info
        self.__logger = LogManager("server.log").logger

    def get_black_screen(self):
        """
        :return:
        """
        black_result_list = []
        temp_video_path = self.video_info_dict.get("temp_video_path", None)
        if temp_video_path:
            out, err = (
                ffmpeg
                    .input(temp_video_path)
                    # pix_th默认为0，10，指黑色像素阈值
                    .filter('blackdetect', d=0.5, pic_th=0.999, pix_th=0.05)
                    .output('pipe:', format='null')
                    .run(quiet=True, capture_stdout=True)
            )
            for i in err.decode().strip().split('\n'):
                if "[blackdetect" in i:
                    try:
                        match_objects = re.match(".*black_start:(.*) black_end:(.*) black_duration:(.*)", i,
                                                 re.M | re.I)
                        black_frames_info = {
                            "black_start": match_objects.group(1),
                            "black_end": match_objects.group(2),
                            "black_duration": match_objects.group(3)
                        }
                        black_result_list.append(black_frames_info)
                    except Exception as err:
                        self.__logger.error(err)
                        continue
        return black_result_list


class DeepVideoIndex(object):
    """ 视觉调用类对外接口类，可以得到视频的所有指标
    """

    def __init__(self, video_info=None):
        self.video_info = video_info
        self.frames_info_dict = {}
        self.__logger = LogManager("server.log").logger

        self.__first_frame_server_url = "http://172.22.119.82:8501/v1/models/first_frame_model:predict"
        self.__start_app_server_url = "http://172.22.119.82:8501/v1/models/start_app_model:predict"
        self.__start_app_tencent_server_url = "http://172.22.119.82:8501/v1/models/start_app_tencent_model:predict"
        self.__start_app_iqiyi_server_url = "http://172.22.119.82:8501/v1/models/start_app_iqiyi_model:predict"
        self.__start_app_youku_server_url = "http://172.22.119.82:8501/v1/models/start_app_youku_model:predict"
        self.__start_app_ixigua_server_url = "http://172.22.119.82:8501/v1/models/start_app_ixigua_model:predict"
        self.__start_app_douyin_server_url = "http://172.22.119.82:8501/v1/models/start_app_douyin_model:predict"
        self.__start_app_comic_server_url = "http://172.22.119.82:8501/v1/models/start_app_comic_model:predict"

        self.__blurred_screen_server_url = ""
        self.__black_screen_server_url = ""
        self.__freeze_screen_server_url = ""
        self.__bfs_url = "http://uat-bfs.bilibili.co/bfs/davinci"  # 上传分帧图片到bfs保存
        self.__task_queue = queue.Queue()
        self.__session = self.__get_http_session(pool_connections=2,
                                                 pool_maxsize=1000,
                                                 max_retries=3
                                                 )

    def __get_video_info(self):
        """
        opencv 获取视频基本信息
        :return:
        """
        cap = cv2.VideoCapture(self.video_info.get("temp_video_path"))
        total_frame = cap.get(7)  # 帧数
        fps = cap.get(5)  # 帧率
        per_frame_time = 1 / fps
        return total_frame, fps, per_frame_time

    def __get_http_session(self, pool_connections, pool_maxsize, max_retries):
        session = requests.Session()
        # 创建一个适配器，连接池的数量pool_connections, 最大数量pool_maxsize, 失败重试的次数max_retries
        adapter = requests.adapters.HTTPAdapter(pool_connections=pool_connections,
                                                pool_maxsize=pool_maxsize,
                                                max_retries=max_retries,
                                                pool_block=True
                                                )
        # 告诉requests，http协议和https协议都使用这个适配器
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    def __upload_frame(self, frame_data):
        """ 上传分帧数据到bfs, 如果超时，就返回空字符串
        :param frame_data:
        :return:
        """
        retry_time = 0
        while retry_time < 3:
            try:
                retry_time += 1
                headers = {"Content-type": "image/png"}
                res = self.__session.put(self.__bfs_url,
                                         data=frame_data,
                                         headers=headers
                                         )
                if res.status_code == 200:
                    return res.headers.get('Location')
                else:
                    res.raise_for_status()
            except Exception as err:
                continue
                # self.__logger.error(err)
                # return ''
        self.__logger.error('bfs > 3')
        return ''

    def __predict_single_frame(self, frame_list, model_server_url):
        """
        :param frame_list: predict_data
        :param model_server_url: model_server
        :return:
        """
        retry_time = 0
        while retry_time < 3:
            try:
                headers = {"content-type": "application/json"}
                body = {"instances": [{"input_1": frame_list}]}
                response = self.__session.post(model_server_url,
                                               data=json.dumps(body),
                                               headers=headers)
                if response.status_code == 200:
                    prediction = response.json()['predictions'][0]
                    return np.argmax(prediction)
                else:
                    response.raise_for_status()
            except Exception as err:
                continue
                # self.__logger.error(err)
                # raise Exception(err)
        self.__logger.error('model > 3')
        raise Exception('model > 3')

    @my_async_decorator
    def __upload_frame_and_cls(self, frame_list=None, frame_data=None, model_type=None):
        """ 上传分帧图片，并调用模型服务，对帧分类
        虽然是异步任务，但是可以把名字一一对应起来,前提是不使用as_completed，
        后续可以把upload_frame任务和frame_cls任务合并成一个异步任务
        :return:
        """
        if model_type == ModelType.STARTAPP:
            model_server_url = self.__start_app_server_url

        elif model_type == ModelType.FIRSTFRAME:
            model_server_url = self.__first_frame_server_url

        elif model_type == ModelType.BLURREDFRAME:
            model_server_url = self.__blurred_screen_server_url

        elif model_type == ModelType.FREEZEFRAME:
            model_server_url = self.__freeze_screen_server_url

        elif model_type == ModelType.BLACKFRAME:
            model_server_url = self.__black_screen_server_url

        elif model_type == ModelType.STARTAPPTENCENT:
            model_server_url = self.__start_app_tencent_server_url

        elif model_type == ModelType.STARTAPPIQIYI:
            model_server_url = self.__start_app_iqiyi_server_url

        elif model_type == ModelType.STARTAPPYOUKU:
            model_server_url = self.__start_app_youku_server_url

        elif model_type == ModelType.STARTAPPIXIGUA:
            model_server_url = self.__start_app_ixigua_server_url

        elif model_type == ModelType.STARTAPPDOUYIN:
            model_server_url = self.__start_app_douyin_server_url

        elif model_type == ModelType.STARTAPPCOMIC:
            model_server_url = self.__start_app_comic_server_url

        else:
            raise Exception("model type is wrong or not supported")

        frame_url = self.__upload_frame(frame_data)
        prediction = self.__predict_single_frame(frame_list, model_server_url)
        del frame_list
        del frame_data
        return prediction, frame_url

    def __cut_frame_upload_predict(self, model_type=None):
        """ 基于opencv的切割图片并上传bfs, 每秒保存7帧，对于人的视觉来看足够
        :return:
        """
        total_frame, fps, per_frame_time = self.__get_video_info()
        cap = cv2.VideoCapture(self.video_info.get("temp_video_path"))
        _, _ = cap.read()
        count = 0
        success = True
        predict_async_tasks = {}
        while success:
            count += 1
            if count % (fps // 10) == 0:
                success, image = cap.read()
                if success:
                    image_col, image_row = image.shape[0], image.shape[1]
                    image = Image.fromarray(image)  # 先转格式为Image 为了统一输入图像尺寸

                    predict_image = image.resize((90, 160), Image.NEAREST)
                    upload_image = image.resize((int(image_row * 0.4), int(image_col * 0.4)), Image.NEAREST)

                    ret, buf = cv2.imencode(".png", np.asarray(upload_image))
                    frame_byte = Image.fromarray(np.uint8(buf)).tobytes()  # 上传bfs数据格式

                    image = np.asarray(predict_image)
                    if model_type == ModelType.STARTAPPCOMIC:
                        frame_list = image.tolist()  # 模型预测数据格式
                    else:
                        frame_list = (image / 255).tolist()
                    try:
                        predict_async_task = self.__upload_frame_and_cls(frame_list, frame_byte, model_type)
                        predict_async_tasks[predict_async_task] = count * per_frame_time
                    except Exception as err:
                        self.__logger.error(err)
                else:
                    continue
            else:
                success, image = cap.read()
                continue

        for predict_async_task in as_completed(predict_async_tasks):
            time_step = predict_async_tasks[predict_async_task]
            try:
                predict_result, frame_name = predict_async_task.result(timeout=20)
            except Exception as err:
                self.__logger.error(err)
                # self.__logger.error(traceback.print_exc())
                continue
            self.frames_info_dict[frame_name] = [time_step, predict_result]
        self.__session.close()

    def get_first_frame_time(self):
        """ 播放器首帧时间
        :return:
        """
        self.__cut_frame_upload_predict(ModelType.FIRSTFRAME)  # 分帧预测并上传帧到bfs，避免本地压力
        first_frame_handler = FirstFrameTimer(frame_info_dict=self.frames_info_dict)
        first_frame_time, cls_results_dict, first_frame_timestamp = first_frame_handler.get_first_frame_time()

        return first_frame_time, cls_results_dict, first_frame_timestamp

    def get_freeze_frame_info(self):
        """
        :return:[{"freeze_start_time": 0,
                  "freeze_duration_time": 0,
                  "freeze_end_time": 0}
                ]
        """
        total_frame, fps, per_frame_time = self.__get_video_info()
        freeze_handler = PlayerFreezeScreenWatcher(self.video_info)
        freeze_result_list = freeze_handler.get_freeze()
        return freeze_result_list

    def get_blurred_frame_rate(self):
        """获取花屏率
        :return:
        """
        blurred_handler = PlayerBlurredScreenWatcher()
        blurred_screen_rate, cls_results_dict = blurred_handler. \
            get_blurred_screen(self.__upload_frame_and_cls(ModelType.BLURREDFRAME))
        return blurred_screen_rate, cls_results_dict

    def get_black_frame_info(self):
        """
        :return:[{
               "black_start": 0,
               "black_end": 0,
               "black_duration": 0
            }]
        """
        black_handler = PlayerBlackScreenWatcher(self.video_info)
        black_screen_list = black_handler.get_black_screen()
        return black_screen_list

    def get_error_rate(self):
        """
        :return:
        """
        return

    def get_app_start_time(self, index_type):
        """ app启动时间耗时
        :return:
        """
        if index_type == ModelType.STARTAPP.name:
            self.__cut_frame_upload_predict(ModelType.STARTAPP)
        elif index_type == ModelType.STARTAPPYOUKU.name:
            self.__cut_frame_upload_predict(ModelType.STARTAPPYOUKU)
        elif index_type == ModelType.STARTAPPTENCENT.name:
            self.__cut_frame_upload_predict(ModelType.STARTAPPTENCENT)
        elif index_type == ModelType.STARTAPPIQIYI.name:
            self.__cut_frame_upload_predict(ModelType.STARTAPPIQIYI)
        elif index_type == ModelType.STARTAPPDOUYIN.name:
            self.__cut_frame_upload_predict(ModelType.STARTAPPDOUYIN)
        elif index_type == ModelType.STARTAPPCOMIC.name:
            self.__cut_frame_upload_predict(ModelType.STARTAPPCOMIC)
        else:
            self.__cut_frame_upload_predict(ModelType.STARTAPPIXIGUA)
        start_app_handler = StartAppTimer(start_app_dict=self.frames_info_dict)
        start_app_time, cls_results_dict, first_frame_timestamp = start_app_handler.get_first_frame_time()
        return start_app_time, cls_results_dict


class VideoSilenceDetector(object):
    """视(音)频静音探测
    """
    DEFAULT_DURATION = 0.1
    DEFAULT_THRESHOLD = -60

    silence_start_re = re.compile(' silence_start: (?P<start>[-,0-9]+(\.?[0-9]*))$')
    silence_end_re = re.compile(' silence_end: (?P<end>[0-9]+(\.?[0-9]*)) ')

    def __init__(self, video_path=None,
                 silence_threshold=DEFAULT_THRESHOLD,
                 silence_duration=DEFAULT_DURATION,
                 start_time=None, end_time=None):
        self.video_path = video_path
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.start_time = start_time
        self.end_time = end_time

    def get_silent_times(self):
        p = subprocess.Popen((ffmpeg
                              .input(self.video_path, )
                              .filter('silencedetect', n='{}dB'.format(self.silence_threshold), d=self.silence_duration)
                              .output('-', format='null')
                              .compile()
                              ) + ['-nostats'],
                             stderr=subprocess.PIPE)
        output = p.communicate()[1].decode('utf-8')
        if p.returncode != 0:
            sys.stderr.write(output)
            sys.exit(1)
        lines = output.splitlines()

        silence_start = []
        silence_end = []
        for line in lines:
            silence_start_match = VideoSilenceDetector.silence_start_re.search(line)
            silence_end_match = VideoSilenceDetector.silence_end_re.search(line)
            if silence_start_match:
                silence_start_tmp = float(silence_start_match.group('start'))
                # 若出现silence_start或silence_end小于0，则需要另外处理下
                silence_start.append(0 if silence_start_tmp < 0. else silence_start_tmp)
            elif silence_end_match:
                silence_end_tmp = float(silence_end_match.group('end'))
                silence_end.append(0 if silence_end_tmp < 0. else silence_end_tmp)
        res = []
        all_duration = ffmpeg.probe(self.video_path)['format']['duration']  # 该视频的总时长
        if len(silence_start) == 1:
            silence_duration = silence_end[0] - silence_start[0]
            # 由于视频的总时长和静音持续时长会存在误差，需要考虑
            if 0.1 >= silence_duration - float(all_duration) >= -0.1:
                silence_info = {
                    "silence_start": silence_start[0],
                    "silence_end": silence_end[0],
                    "silence_duration": silence_duration
                }
                res.append(silence_info)
        return res


class VideoColourCastDetector(object):
    """色差检测"""

    def __init__(self, video_path=None, src_video_path=None, target_video_path=None):
        self.video_path = video_path
        self.src_video_path = src_video_path
        self.target_video_path = target_video_path
        self.video_color_primaries = []

    def get_space_info(self):
        """获取视频色彩空间信息，判断视频是否会偏色
        """
        info_dict = ffmpeg.probe(self.video_path)
        info_list = info_dict['streams']
        for info in info_list:
            if 'color_primaries' in info:
                self.video_color_primaries.append(info['color_primaries'])
        if len(self.video_color_primaries) > 0:
            if self.video_color_primaries[0] == 'bt2020':
                return {"judge": True, "color_primaries": self.video_color_primaries[0]}
            else:
                return {"judge": False, "color_primaries": self.video_color_primaries[0]}
        else:
            return {"judge": False, "color_primaries": None}

    def get_average_chroma(self):
        """根据相同帧数的色度平均值判断是否偏色
        """
        src_cap = cv2.VideoCapture(self.src_video_path)
        src_frame_number = src_cap.get(7)
        # 截取2——3帧
        rate = int(src_frame_number / 3)
        src_chroma_list = self.__get_video_frame(self.src_video_path, rate)
        target_chroma_list = self.__get_video_frame(self.target_video_path, rate)
        if len(src_chroma_list) == len(target_chroma_list):
            for i in range(len(src_chroma_list)):
                src_chroma = round(src_chroma_list[i], 2)
                target_chroma = round(target_chroma_list[i], 2)
                k = abs(src_chroma - target_chroma) / max(src_chroma, target_chroma)
                if k >= 0.05:
                    return {"judge": True}
                else:
                    continue
            return {"judge": False}
        else:
            for i in range(min(len(src_chroma_list), len(target_chroma_list))):
                src_chroma = round(src_chroma_list[i], 2)
                target_chroma = round(target_chroma_list[i], 2)
                k = abs(src_chroma - target_chroma) / max(src_chroma, target_chroma)
                if k >= 0.05:
                    return {"judge": True}
                else:
                    continue
            return {"judge": False}

    def __count_average_chroma(self, img):
        """计算该帧的色度平均值
        """
        m, n, c = img.shape
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(img_lab)
        d_a, d_b = 0, 0
        for i in range(m):
            for j in range(n):
                d_a = d_a + a[i][j]
                d_b = d_b + b[i][j]
        pixel_sum = m * n
        d_a, d_b = (d_a / pixel_sum) - 128, (d_b / pixel_sum) - 128
        d = np.sqrt((np.square(d_a) + np.square(d_b)))
        return d

    def __get_video_frame(self, video_path, rate):
        """截帧
        """
        chroma_list = []
        cap = cv2.VideoCapture(video_path)
        c = 1
        while True:
            ret, frame = cap.read()
            if ret:
                if c % rate == 0:
                    chroma = self.__count_average_chroma(frame)
                    chroma_list.append(chroma)
                c += 1
            else:
                break
        cap.release()
        return chroma_list


if __name__ == '__main__':
    # cv_info = {"temp_video_path": '/Users/luoyadong/Desktop/video.mp4'}
    # deep_index_handler = DeepVideoIndex(cv_info)
    # print(json.dumps(deep_index_handler.get_app_start_time("STARTAPPYOUKU")))
    # first_frame_time, cls_results_dict = deep_index_handler.get_first_frame_time()
    # freeze_frame_list = deep_index_handler.get_freeze_frame_info()
    # black_frame_list = deep_index_handler.get_black_frame_info()
    videosilence = VideoSilenceDetector(video_path="/Users/bilibili/Downloads/这一不小心就当.mp4")
    print(videosilence.get_silent_times())
    print(ffmpeg.probe("/Users/bilibili/Downloads/这一不小心就当.mp4")['format']['duration'])
    # print(PlayerBlackScreenWatcher(video_info={'temp_video_path':'/Users/bilibili/Desktop/26899393.mp4'}).get_black_screen())
