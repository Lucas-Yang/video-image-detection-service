"""
模型预测封装类
通过调用该类获取视频播放质量指标
"""
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import json
import os
import requests
import numpy as np
import queue

from collections import OrderedDict
from PIL import Image
from io import BytesIO
from enum import Enum

from functools import wraps
from app.factory import LogManager, MyThread

thread_executor = ThreadPoolExecutor(max_workers=100)


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
        :param stage_name_list: 各个阶段的别名，用于生成报告
        """

        if stage_name_list is None:
            stage_name_list = ["阶段0: 播放器打开", "阶段1: 播放器加载", "阶段2: 播放器播放", "阶段3: 无关阶段"]
        self.frame_info_dict = frame_info_dict  # frame时间戳映射表
        self.first_frame_time = 0
        self.stage_name_list = stage_name_list
        self._cls_dict = {}

    def _sort_reformat_result(self):
        """ 排序整理好后的分类dict
        :return:
        """
        for key, value in self._cls_dict.items():
            value.sort(key=lambda x: self.frame_info_dict[x])

    def _reformat_result(self, predict_image_json):
        """ 将整个视频图片的预测结构分类整理成一个dict
        :return:
        """
        if int(list(predict_image_json.values())[0]) in self._cls_dict:
            self._cls_dict[int(list(predict_image_json.values())[0])].append(list(predict_image_json.keys())[0])
        else:
            self._cls_dict[int(list(predict_image_json.values())[0])] = [list(predict_image_json.keys())[0]]

    def get_first_frame_time(self, predict_result_list):
        """
        :return:
        """
        for predict_image_json in predict_result_list:
            self._reformat_result(predict_image_json)
        self._sort_reformat_result()
        # print("排序结果: ", self._cls_dict)
        # 正常逻辑，每个阶段都正常识别
        if len(self._cls_dict.get(0, [])) and len(self._cls_dict.get(2, [])):
            self.first_frame_time = float(self.frame_info_dict.get(self._cls_dict.get(2)[0])) - \
                                    float(self.frame_info_dict.get(self._cls_dict.get(0)[0]))
        # 当播放完成阶段没有时候，返回-1,给上层判断
        else:
            self.first_frame_time = -1

        ordered_dict = OrderedDict(sorted(self._cls_dict.items(), key=lambda obj: obj[0]))
        return self.first_frame_time, ordered_dict


class StartAppTimer(FirstFrameTimer):
    """ 启动时间计算类
    """

    def __init__(self):
        self.stage_name_list = ["阶段0：app打开", "阶段1：app推荐页加载", "阶段2：app正确启动页面", "阶段3：其他无关页面"]
        super.__init__()


class PlayerFreezeScreenWatcher(object):
    """ 卡顿计算类
    """

    def __init__(self):
        pass

    def get_freeze(self, predict_result_list: list):
        """
        :param predict_result_list:
        :return:
        """
        return None, None


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

    def __init__(self):
        pass

    def get_black_screen(self, predict_result_list: list):
        """
        :param predict_result_list:
        :return:
        """
        return None, None


class DeepVideoIndex(object):
    """ 视觉调用类对外接口类，可以得到视频的所有指标
    """
    def __init__(self, video_info=None):
        self.video_info = video_info
        self.frames_info_dict = {}
        self.__logger = LogManager("server.log").logger

        self.__first_frame_server_url = "http://172.16.60.71:8501/v1/models/first_frame_model:predict"
        self.__start_app_server_url = "http://172.16.60.71:8501/v1/models/start_app_model:predict"
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

    @my_async_decorator
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

    def __cut_frame_upload(self):
        """ 基于opencv的切割图片并上传bfs, 每秒保存10帧，对于人的视觉来看足够
        :return:
        """
        total_frame, fps, per_frame_time = self.__get_video_info()
        cap = cv2.VideoCapture(self.video_info.get("temp_video_path"))
        success, image = cap.read()
        os.remove(self.video_info.get("temp_video_path"))  # 删除临时视频文件
        count = 0
        success = True
        async_tasks = []
        while success:
            count += 1
            if count % (fps // 10) == 0:
                success, image = cap.read()

                ret, buf = cv2.imencode(".png", image)
                frame_byte = Image.fromarray(np.uint8(buf)).tobytes()
                # frame_list = (image / 255).tolist()
                # print(self.__frame_cls(model_type=ModelType.FIRSTFRAME, image_list=frame_list))
                # break
                try:
                    frame_async = self.__upload_frame(frame_byte)
                    async_tasks.append([frame_async, count * per_frame_time])
                except Exception as err:
                    self.__logger.error(err)
                if count > 100:
                    break
            else:
                success, image = cap.read()
                continue

        for async_task in async_tasks:
            frame_name = async_task[0].result()
            self.frames_info_dict[frame_name] = async_task[1]

        # 实验
        # for key, _ in self.frames_info_dict.items():
        #    print("0000", key)
        #    break

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

    @my_async_decorator
    def __frame_cls(self, image_url: str = None, model_type=None, image_list=None):
        """ 调用模型服务，对帧分类
        暂时没法直接用分帧的数据直接预测，因为bfs的文件名是不确定的，且上传文件是一个异步任务，没法一一对应起来，
        后续自定义上传文件名，可以一边上传一边预测，可节省一半的时间
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

        headers = {"content-type": "application/json"}
        if image_list:
            body = {"instances": [{"input_1": image_list}]}
        else:
            body = {"instances": [{"input_1": self.__load_image_url(image_url=image_url)}]}

        response = requests.post(model_server_url, data=json.dumps(body), headers=headers)
        response.raise_for_status()
        prediction = response.json()['predictions'][0]
        return {image_url: np.argmax(prediction)}

    def __video_predict(self, model_type: ModelType):
        """ 将视频分帧，上传到bfs，再对所有的帧进行分类,
        调用self.__frame_cls 是异步任务，所以第二个for循环是用于获取结果
        :return: 所有帧的分类结果 [{cls: image_url}]
        """
        cls_result_list = []
        async_tasks = []
        for frame_data_url, _ in self.frames_info_dict.items():
            cls_result_async = self.__frame_cls(frame_data_url, model_type)
            async_tasks.append(cls_result_async)

        for async_task in as_completed(async_tasks):
            cls_result = async_task.result()
            cls_result_list.append(cls_result)
            # del async_tasks[async_task]
        # print("1111", cls_result_list[0])
        return cls_result_list

    def get_first_video_time(self):
        """ 播放器首帧时间
        :return:
        """
        self.__cut_frame_upload()  # 分帧上传帧到bfs，避免本地压力
        predict_result_list = self.__video_predict(ModelType.FIRSTFRAME)  # 拉取图片预测
        first_frame_handler = FirstFrameTimer(frame_info_dict=self.frames_info_dict)
        first_frame_time, cls_results_dict = first_frame_handler. \
            get_first_frame_time(predict_result_list=predict_result_list)

        return first_frame_time, cls_results_dict

    def get_freeze_rate(self):
        """
        :return:
        """
        freeze_handler = PlayerFreezeScreenWatcher()
        freeze_rate, cls_results_dict = freeze_handler. \
            get_freeze(self.__video_predict(ModelType.FREEZESREEN))
        return freeze_rate, cls_results_dict

    def get_blurred_screen_rate(self):
        """获取花屏率
        :return:
        """
        blurred_handler = PlayerBlurredScreenWatcher()
        blurred_screen_rate, cls_results_dict = blurred_handler. \
            get_blurred_screen(self.__video_predict(ModelType.BLURREDSCREEN))
        return blurred_screen_rate, cls_results_dict

    def get_black_screen_rate(self):
        """
        :return:
        """
        black_handler = PlayerBlackScreenWatcher()
        black_screen_rate, cls_results_dict = black_handler. \
            get_black_screen(self.__video_predict(ModelType.BLACKSCREEN))
        return black_screen_rate, cls_results_dict

    def get_error_rate(self):
        """
        :return:
        """
        return

    def get_app_start_time(self):
        """ app启动时间耗时
        :return:
        """
        start_app_handler = StartAppTimer()
        start_app_time, cls_results_dict = start_app_handler. \
            get_first_frame_time(self.__video_predict(ModelType.STARTAPP))
        return start_app_time, cls_results_dict
