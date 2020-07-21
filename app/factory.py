"""
工具文件：封装各种操作类，例如日志，数据库操作，缓存等等
"""

import logging
import os
import cloghandler


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

        self.__stream_handler = None
        self.__file_handler = None
        self.get_logger()

    def __ini_handler(self):
        """
        :return:
        """
        self.__stream_handler = logging.StreamHandler()
        self.__file_handler = cloghandler.ConcurrentRotatingFileHandler(self.__path, "a", 1024 * 1024 * 100, 5,
                                                                        encoding='utf-8')

    def __set_handler(self,  level='DEBUG'):
        """
        :param level:
        :return:
        """
        self.__stream_handler.setLevel(level)
        self.__file_handler.setLevel(level)
        self.logger.addHandler(self.__stream_handler)
        self.logger.addHandler(self.__file_handler)

    def __set_formatter(self):
        """
        :return:
        """
        formatter = logging.Formatter('%(asctime)s-%(name)s-%(filename)s-[line:%(lineno)d]'
                                      '-%(levelname)s: %(message)s',
                                      datefmt='%a, %d %b %Y %H:%M:%S')
        self.__stream_handler.setFormatter(formatter)
        self.__file_handler.setFormatter(formatter)

    def __close_handler(self):
        """
        :return:
        """
        self.__stream_handler.close()
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


if __name__ == "__main__":
    log = LogManager("test.log").logger
    for i in range(10000):
        log.info("11111")
