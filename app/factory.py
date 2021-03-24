"""
工具文件：封装各种操作类，例如日志，数据库操作封装，缓存，接口输入参数校验等等
"""

import os
import cloghandler
import json
import logging
import time
import threading
from jsonschema import validate
from pymongo import MongoClient
from app.data import VideoQualityItem

MONGO_URL = 'mongodb://burytest:GbnO35lpzAyjkPqSXQTiHwLuDs2r4gcR@172.22.34.102:3301/test' \
            '?authSource=burytest&replicaSet=bapi&readPreference=primary&appname=MongoDB%2' \
            '0Compass&ssl=false'
MONGO_DB = 'burytest'


class LogManager(object):
    """
    # 日志封装类，加文件锁避免了多进程写不安全情况
    # 日志分块，每个文件最大100M，保留5个文件
    """

    def __init__(self, path='server.log', level='DEBUG', name=__name__):
        """
        :param path:
        :param level:
        :param name:
        """
        cur_path = os.path.dirname(os.path.realpath(__file__))
        log_path = os.path.join(os.path.dirname(cur_path), 'log')
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        self.__path = os.path.join(log_path, path)
        self.__name = name
        self.__level = level
        self.logger = logging.getLogger(self.__name)
        self.logger.setLevel(self.__level)

        self.stream_handler = None
        self.file_handler = None
        self.get_logger()

    def __ini_handler(self):
        """
        :return:
        """
        self.stream_handler = logging.StreamHandler()
        self.file_handler = cloghandler.ConcurrentRotatingFileHandler(self.__path, "a", 1024 * 1024 * 100, 5,
                                                                        encoding='utf-8')

    def __set_handler(self, level='DEBUG'):
        """
        :param level:
        :return:
        """
        if not self.logger.handlers:
            self.stream_handler.setLevel(level)
            self.file_handler.setLevel(level)
            self.logger.addHandler(self.stream_handler)
            self.logger.addHandler(self.file_handler)

    def __set_formatter(self):
        """
        :return:
        """
        formatter = logging.Formatter('%(asctime)s-%(name)s-%(filename)s-[line:%(lineno)d]'
                                      '-%(levelname)s: %(message)s',
                                      datefmt='%a, %d %b %Y %H:%M:%S')
        self.stream_handler.setFormatter(formatter)
        self.file_handler.setFormatter(formatter)

    def __close_handler(self):
        """
        :return:
        """
        self.stream_handler.close()
        # self.__file_handler.close()

    def get_logger(self):
        """
        :return:
        """
        self.__ini_handler()
        self.__set_formatter()
        self.__set_handler()
        self.__close_handler()
        # return self.logger


class FormatChecker(object):
    """ 输入参数校验类，所有接口的输入参数校验都放在该类中
    """

    def __init__(self):
        self.__logger = LogManager("server.log").logger

    def player_index_dot_check(self, input_json):
        """ 打点指标获取接口 输入参数校验
        :param input_json:
        :return:
        """
        json_schema = {
            "type": "object",
            "requiredv": True,
            "properties": {
                "device_id": {
                    "type": "string",
                    "minlength": 2
                },
                "buvid": {
                    "type": "string",
                    "minlength": 2
                },
                "start_time": {
                    "type": "string",
                    "minlength": 2
                },
                "end_time": {
                    "type": "string",
                    "minlength": 2
                }
            },
            "required": [
                "device_id",
                "buvid",
                "start_time",
                "end_time"
            ]
        }
        try:
            input_json = json.loads(input_json)
            validate(input_json, json_schema)
        except BaseException as err:
            self.__logger.error(err)
            return False
        return True

    def player_index_cv_check(self, request):
        """
        :param request:
        :return:
        """
        json_schema = {
            "type": "object",
            "requiredv": True,
            "properties": {
                "index_types": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 5,
                    "items": {
                        "type": "string",
                        "enum": ["FIRSTFRAME", "STARTAPP", "BLACKFRAME", "BLURREDFRAME", "FREEZEFRAME",
                                 "STARTAPPYOUKU", "STARTAPPIXIGUA", "STARTAPPTENCENT", "STARTAPPIQIYI",
                                 "STARTAPPDOUYIN", "STARTAPPCOMIC"
                                 ]
                    }
                }
            },
            "required": [
                "index_types"
            ]
        }
        try:
            input_json = {"index_types": request.form.getlist("index_types")}
            validate(input_json, json_schema)
            if request.files['file'].filename.split('.')[-1] != "mp4" and \
                    request.files['file'].filename.split('.')[-1] != "MP4":
                raise Exception("input file is not mp4")
        except BaseException as err:
            self.__logger.error(err)
            return False
        return True

    def video_index_cv_check(self, index_types, file):
        """ fastapi upload接口参数校验
        :param index_types:
        :param file:
        :return:
        """
        for index_type in index_types:
            if index_type not in VideoQualityItem.__members__:
                raise Exception("{} is wrong, plz make sure item in {}".
                                format(index_type,
                                       [index_type for index_type, _ in VideoQualityItem.__members__.items()]
                                       )
                                )
            else:
                continue
        if not (file.filename.endswith('.mp4') or file.filename.endswith('.MP4')):
            self.__logger.error(file.content_type)
            raise Exception("input error, {} is not a video file".format(file.filename))
        else:
            return True

    def ssim_index_checker(self, request):
        """ 相似性计算文档
        :param request:
        :return:
        """
        try:
            if request.files['file_src'].filename.split('.')[-1] != "mp4" and \
                    request.files['file_src'].filename.split('.')[-1] != "MP4":
                raise Exception("input file is not mp4")
            if request.files['file_target'].filename.split('.')[-1] != "mp4" and \
                    request.files['file_target'].filename.split('.')[-1] != "MP4":
                raise Exception("input file is not mp4")
        except BaseException as err:
            self.__logger.error(err)
            return False
        return True

    def silence_index_checker(self, filename):
        """ 静音文件格式检查
        :param request:
        :return:
        """
        allowed_formats = ('mp4', 'MP4', 'mkv', 'MKV', 'mp3', 'MP3', 'aac', 'AAC', 'wav', 'WAV')
        try:
            if filename.split('.')[-1] not in allowed_formats:
                raise Exception("the format of the input file is not accepted")
        except BaseException as err:
            self.__logger.error(err)
            return False
        return True

    def api_ssim_index_checker(self, file):
        """
        :param file:
        :return:
        """
        if file.content_type != 'video/mp4':
            raise Exception("input error, {} is not a video file".format(file.filename))
        else:
            return True

    def image_white_detection_checker(self, request_body):
        """ 图像检测-校验
        :param request_body:
        :return:
        """
        try:
            if request_body.files['file'].filename.split('.')[-1] != "png" and \
                    request_body.files['file'].filename.split('.')[-1] != "jpg":
                raise Exception("input file is not png/jpg")
        except BaseException as err:
            self.__logger.error(err)
            return False
        return True

    def api_image_white_detection_checker(self, file_image):
        """ 图像检测-校验
        :param file_image:
        :return:
        """
        try:
            if 'image' not in file_image.content_type:
                raise Exception("input file is not png/jpg")
        except BaseException as err:
            self.__logger.error(err)
            return False
        return True


class MyMongoClient(object):
    """mongo 操作类
    """
    def __init__(self):
        client = MongoClient(MONGO_URL)
        self.db = client.get_database(MONGO_DB)
        self.db_initial_time = time.strftime("%Y-%m-%d %H:%M:%S")

    def db(self):
        """
        :return:
        """
        return self.db

    def insert(self, collection, data):
        """
        :param collection:
        :param data:
        :return:
        """
        if not isinstance(data, dict):
            raise Exception("args `data` is not dict, plz check!")
        data['create_time'] = self.db_initial_time
        try:
            self.db.get_collection(collection).insert_one(data)
        except Exception as err:
            if "duplicate key error collection" in str(err):
                return
            else:
                raise Exception(err)

    def query(self, collection, select=None):
        """
        :param collection:
        :param select:
        :return:
        """
        return self.db.get_collection(collection).find(select, no_cursor_timeout=True)


class MyThread(threading.Thread):
    """适配装饰器，重写Thread，新增get_result函数用于获取执行多线程函数的return值
    """
    def __init__(self, func, args=(), kwargs=None):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args
        self.__logger = LogManager("server.log").logger
        self.result = None

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        threading.Thread.join(self)
        try:
            return self.result  # 如果子线程不使用join方法，此处可能会报没有self.result的错误
        except Exception as err:
            self.__logger.error(err)
            return None


if __name__ == "__main__":
    db_handler = MyMongoClient()
    db_handler.insert("video_parsing_tasks", {"task_id": "be0e768e-4034-427e-8dee-06ed1c0dc60test1",
                                              "task_result": 11})
    result = db_handler.query("video_parsing_tasks", {"task_id": "be0e768e-4034-427e-8dee-06ed1c0dc60test1"}).sort("create_time", -1).limit(1)

    for data in result:
        print(data.get("task_result", None))
