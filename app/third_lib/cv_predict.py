"""
模型预测封装类
通过调用该类获取视频播放质量指标
"""
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import cv2
import ffmpeg
import json
import os
import re
import requests
import queue

from collections import OrderedDict
import numpy as np
from PIL import Image
from io import BytesIO
from enum import Enum

from functools import wraps
from app.factory import LogManager, MyThread

thread_executor = ThreadPoolExecutor(max_workers=20)


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
    FREEZESREEN = 3  # 卡顿
    BLURREDSCREEN = 4  # 花屏
    BLACKSCREEN = 5  # 黑屏


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
        # 正常逻辑，每个阶段都正常识别
        if len(self._cls_dict.get(0, [])) and len(self._cls_dict.get(2, [])):
            self.first_frame_time = float(self.frame_info_dict.get(self._cls_dict.get(2)[0][0])[0]) - \
                                    float(self.frame_info_dict.get(self._cls_dict.get(0)[0][0])[0])
        # 当播放完成阶段没有时候，返回-1,给上层判断
        elif len(self._cls_dict.get(1, [])) and len(self._cls_dict.get(2, [])):
            self.first_frame_time = float(self.frame_info_dict.get(self._cls_dict.get(2)[0][0])[0]) - \
                                    float(self.frame_info_dict.get(self._cls_dict.get(1)[0][0])[0])
        else:
            self.first_frame_time = -1

        ordered_dict = OrderedDict(sorted(self._cls_dict.items(), key=lambda obj: obj[0]))
        return self.first_frame_time, ordered_dict


class StartAppTimer(FirstFrameTimer):
    """ 启动时间计算类
    """

    def __init__(self, start_app_dict=None):
        self.stage_name_list = ["阶段0：app打开", "阶段1：app推荐页加载", "阶段2：app正确启动页面", "阶段3：其他无关页面"]
        super.__init__()


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
            stream = ffmpeg.input('/Users/luoyadong/Desktop/test1.mp4')
            stream = ffmpeg.filter_(stream, 'freezedetect', n=0.001, d=0.3)
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
                            # print("freeze_start", match_objects_start.group(1))
                        elif "freeze_duration" in line_info:
                            match_objects_duration = re.match(".*freeze_duration: (.*)", line_info, re.M | re.I)
                            freeze_duration_list.append(match_objects_duration.group(1))
                            # print("freeze_duration", match_objects_duration.group(1))
                        elif "freeze_end" in line_info:
                            match_objects_duration = re.match(".*freeze_end: (.*)", line_info, re.M | re.I)
                            freeze_end_list.append(match_objects_duration.group(1))
                            # print("freeze_end", match_objects_duration.group(1))
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
                    .filter('blackdetect', d=0.5, pic_th=0.8)
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

        self.__first_frame_server_url = "http://10.217.16.154:8501/v1/models/first_frame_model:predict"
        self.__start_app_server_url = "http://10.217.16.154:8501/v1/models/start_app_model:predict"
        self.__blurred_screen_server_url = ""
        self.__black_screen_server_url = ""
        self.__freeze_screen_server_url = ""
        self.__bfs_url = "http://uat-bfs.bilibili.co/bfs/davinci"  # 上传分帧图片到bfs保存
        self.__task_queue = queue.Queue()

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

    def __upload_frame(self, frame_data):
        """ 上传分帧数据到bfs
        :param frame_data:
        :return:
        """
        try_time = 0
        while try_time < 3:
            try:
                headers = {"Content-type": "image/png"}
                res = requests.put(self.__bfs_url, data=frame_data, headers=headers)
                if res.status_code == 200:
                    return res.headers.get('Location')
                else:
                    res.raise_for_status()
            except requests.exceptions.RequestException:
                try_time += 1
        raise Exception("access bfs error time > 3")

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
        elif model_type == ModelType.BLURREDSCREEN:
            model_server_url = self.__blurred_screen_server_url
        elif model_type == ModelType.FREEZESREEN:
            model_server_url = self.__freeze_screen_server_url
        elif model_type == ModelType.BLACKSCREEN:
            model_server_url = self.__black_screen_server_url
        else:
            raise Exception("model type is wrong or not supported")
        frame_url = self.__upload_frame(frame_data)

        headers = {"content-type": "application/json"}
        body = {"instances": [{"input_1": frame_list}]}
        response = requests.post(model_server_url, data=json.dumps(body), headers=headers)
        response.raise_for_status()
        prediction = response.json()['predictions'][0]
        del frame_list
        del frame_data
        # print("{}, {}".format(str(np.argmax(prediction)), frame_url))
        return np.argmax(prediction), frame_url

    def __cut_frame_upload_predict(self, model_type=None):
        """ 基于opencv的切割图片并上传bfs, 每秒保存10帧，对于人的视觉来看足够
        :return:
        """
        total_frame, fps, per_frame_time = self.__get_video_info()
        cap = cv2.VideoCapture(self.video_info.get("temp_video_path"))
        _, _ = cap.read()
        # os.remove(self.video_info.get("temp_video_path"))  # 删除临时视频文件
        count = 0
        success = True
        predict_async_tasks = {}
        while success:
            count += 1
            if count % (fps // 5) == 0:
                success, image = cap.read()

                image_col, image_row = image.shape[0], image.shape[1]
                image = Image.fromarray(image)  # 先转格式为Image 为了统一输入图像尺寸

                predict_image = image.resize((90, 160), Image.NEAREST)
                upload_image = image.resize((int(image_row * 0.4), int(image_col * 0.4)), Image.NEAREST)

                ret, buf = cv2.imencode(".png", np.asarray(upload_image))
                frame_byte = Image.fromarray(np.uint8(buf)).tobytes()  # 上传bfs数据格式

                image = np.asarray(predict_image)
                frame_list = (image / 255).tolist()  # 模型预测数据格式
                try:
                    predict_async_task = self.__upload_frame_and_cls(frame_list, frame_byte, model_type)
                    predict_async_tasks[predict_async_task] = count * per_frame_time
                except Exception as err:
                    self.__logger.error(err)
                # 实验
                # if count > 10:
                #     break
            else:
                success, image = cap.read()
                continue

        for predict_async_task in as_completed(predict_async_tasks):
            time_step = predict_async_tasks[predict_async_task]
            try:
                predict_result, frame_name = predict_async_task.result(timeout=20)
            except Exception as err:
                self.__logger.error(err)
                continue
            self.frames_info_dict[frame_name] = [time_step, predict_result]

    @classmethod
    def __load_image_url(cls, image_url: str):
        """ 从url 读取图片数据, 返回多位数组(模型服务接口输入是数据，不是np)
        :param image_url:
        :return:
        """
        try_time = 0
        while try_time < 3:
            try:
                img = Image.open(BytesIO(requests.get(image_url).content))
                img = img.convert('RGB')
                img = img.resize((160, 90), Image.NEAREST)
                img = np.asarray(img)
                img = img / 255  # 此处还需要将0-255转化为0-1
                img = img.tolist()
                return img
            except Exception as err:
                try_time += 1
        raise Exception("access bfs error time > 3")

    def get_first_frame_time(self):
        """ 播放器首帧时间
        :return:
        """
        self.__cut_frame_upload_predict(ModelType.FIRSTFRAME)  # 分帧预测并上传帧到bfs，避免本地压力
        # print(self.frames_info_dict)
        first_frame_handler = FirstFrameTimer(frame_info_dict=self.frames_info_dict)
        first_frame_time, cls_results_dict = first_frame_handler.get_first_frame_time()

        return first_frame_time, cls_results_dict

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
            get_blurred_screen(self.__video_predict(ModelType.BLURREDSCREEN))
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

    def get_app_start_time(self):
        """ app启动时间耗时
        :return:
        """
        self.__cut_frame_upload_predict(ModelType.STARTAPP)
        start_app_handler = StartAppTimer(start_app_dict=self.frames_info_dict)
        start_app_time, cls_results_dict = start_app_handler.get_first_frame_time()
        return start_app_time, cls_results_dict
