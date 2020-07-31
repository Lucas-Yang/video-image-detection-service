"""
模型预测封装类
通过调用该类获取视频播放质量指标
"""
import json
import re
import requests
from collections import OrderedDict
import numpy as np
from PIL import Image
from io import BytesIO
from enum import Enum


class ModelType(Enum):
    """模型调用枚举，目前暂定支持五种模型，分别调用不同的模型服务
    """
    FIRSTFRAME = 1   # 播放首帧
    STARTAPP = 2   # app启动时间
    FREEZESREEN = 3  # 卡顿
    BLURREDSCREEN = 4  # 花屏
    BLACKSCREEN = 5  # 黑屏


class FirstFrameTimer(object):
    """首帧获取类
    """
    def __init__(self,
                 stage_name_list: list = None
                 ):
        """
        :param stage_name_list: 各个阶段的别名，用于生成报告
        """
        if stage_name_list is None:
            stage_name_list = ["阶段0: 播放器打开", "阶段1: 播放器加载", "阶段2: 播放器播放", "阶段3: 无关阶段"]
        self.first_frame_time = 0
        self.stage_name_list = stage_name_list
        self._cls_dict = {}

    def _sort_reformat_result(self):
        """ 排序整理好后的分类dict
        :return:
        """
        for key, value in self._cls_dict.items():
            value.sort(key=lambda x: int(re.match(r'.*\/screen_(.*)_.*', x, re.M | re.I).group(1)))

    def _reformat_result(self, predict_image_json):
        """ 将整个视频图片的预测结构分类整理成一个dict
        :return:
        """
        if list(predict_image_json.values())[0] in self._cls_dict:
            self._cls_dict[list(predict_image_json.values())[0]].append(list(predict_image_json.keys())[0])
        else:
            self._cls_dict[list(predict_image_json.values())[0]] = [list(predict_image_json.keys())[0]]

    def get_first_frame_time(self, predict_result_list):
        """
        :return:
        """
        for predict_image_json in predict_result_list:
            self._reformat_result(predict_image_json)
        self._sort_reformat_result()
        # print(json.dumps(self.__cls_dict, ensure_ascii=False))
        # 正常逻辑，每个阶段都正常识别
        if len(self._cls_dict.get(0, [])) and len(self._cls_dict.get(2, [])):
            self.first_frame_time = float(re.match(r'.*\/screen_.*_(.*).png', self._cls_dict[2][0],
                                                   re.M | re.I).group(1)) - \
                                    float(re.match(r'.*\/screen_.*_(.*).png', self._cls_dict[0][0],
                                                   re.M | re.I).group(1))
        # 当播放完成阶段没有时候，返回-1,给上层判断
        elif len(self._cls_dict.get(2, [])) == 0 and len(self._cls_dict.get(1, [])) > 0:
            self.first_frame_time = -1
        else:
            pass
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
    """
    """
    def __init__(self, video_info=None):
        self.video_info = video_info

        self.__first_frame_server_url = "http://172.16.60.71:8501/v1/models/first_frame_model:predict"
        self.__start_app_server_url = "http://172.16.60.71:8501/v1/models/start_app_model:predict"
        self.__blurred_screen_server_url = ""
        self.__black_screen_server_url = ""
        self.__freeze_screen_server_url = ""
        self.__bfs_url = "http://uat-bfs.bilibili.co/bfs/davinci"  # 上传分帧图片到bfs保存

    def __upload_frame(self, frame_data):
        """ 上传分帧数据到bfs
        :param frame_data:
        :return:
        """
        headers = {"Content-type": "image/png"}
        res = requests.put(self.__bfs_url, data=frame_data, headers=headers)
        if res.status_code == 200:
            return res.headers.get('Location')
        else:
            res.raise_for_status()

    @staticmethod
    def __load_image_url(image_url: str):
        """ 从url 读取图片数据, 返回多位数组(模型服务接口输入是数据，不是np)
        :param image_url:
        :return:
        """
        img = Image.open(BytesIO(requests.get(image_url).content))
        img = img.convert('RGB')
        img = img.resize((160, 90), Image.NEAREST)
        img = np.asarray(img)
        img = img / 255  # 此处还需要将0-255转化为0-1
        img = img.tolist()
        return img

    def __frame_cls(self, image_url: str, model_type):
        """ 调用模型服务，对帧分类
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
            model_server_url = ModelType.BLACKSCREEN
        else:
            raise Exception("model type is wrong or not supported")

        headers = {"content-type": "application/json"}
        body = {"instances": [{"input_1": self.__load_image_url(image_url)}]}
        response = requests.post(model_server_url, data=json.dumps(body), headers=headers)
        response.raise_for_status()
        prediction = response.json()['predictions'][0]
        return {image_url: np.argmax(prediction)}

    def __video_predict(self, model_type: ModelType):
        """ 将视频分帧，上传到bfs，再对所有的帧进行分类
        :return: 所有帧的分类结果 [{cls: image_url}]
        """
        video_frame_list = []
        cls_result_list = []
        for frame_data in video_frame_list:
            cls_result_list.append(self.__frame_cls(self.__upload_frame(frame_data), model_type=model_type))
        return cls_result_list

    def get_first_video_time(self):
        """
        :return:
        """
        first_frame_handler = FirstFrameTimer()
        first_frame_time, cls_results_dict = first_frame_handler.\
            get_first_frame_time(self.__video_predict(ModelType.FIRSTFRAME))
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
        return

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
