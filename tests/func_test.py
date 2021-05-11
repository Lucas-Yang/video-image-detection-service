"""
重要类函数单元测试
"""

import os

import cv2
import pytest

from app.model import ImageIndex


class TestFunc(object):
    """ 函数接口测试类
    """
    module_path = os.path.dirname(__file__)

    # 函数必须以test_打头
    def test_black_white_frame_detection_true(self):
        filename = self.module_path + '/image_data/white.jpg'
        img = cv2.imread(filename)
        img_bytes = cv2.imencode('.jpg', img)[1]
        a = ImageIndex(img_bytes)
        assert a.black_white_frame_detection() is True

    def test_black_white_frame_detection_false(self):
        filename = self.module_path + '/image_data/normal.jpg'
        img = cv2.imread(filename)
        img_bytes = cv2.imencode('.jpg', img)[1]
        a = ImageIndex(img_bytes)
        assert a.black_white_frame_detection() is False

    def test_blurred_frame_detection_false(self):
        filename = self.module_path + '/image_data/white.jpg'
        img = cv2.imread(filename)
        img_bytes = cv2.imencode('.jpg', img)[1]
        a = ImageIndex(img_bytes)
        assert a.blurred_frame_detection() is False

    def test_blurred_frame_detection_zero(self):
        filename = self.module_path + '/image_data/normal.jpg'
        img = cv2.imread(filename)
        img_bytes = cv2.imencode('.jpg', img)[1]
        a = ImageIndex(img_bytes)
        assert a.blurred_frame_detection() is True

    # -1 是 timeout
    def test_blurred_frame_detection_minus(self):
        filename = self.module_path + '/image_data/cate1.jpg'
        img = cv2.imread(filename)
        img_bytes = cv2.imencode('.jpg', img)[1]
        a = ImageIndex(img_bytes)
        assert a.blurred_frame_detection() == -1

    def test_frame_ocr(self):
        filename = self.module_path + '/image_data/ocr.jpg'
        img = cv2.imread(filename)
        img_bytes = cv2.imencode('.jpg', img)[1]
        a = ImageIndex(img_bytes)
        assert a.frame_ocr() is not None

    def test_color_layer(self):
        filename = self.module_path + '/image_data/color_layer_detect1.png'
        img = cv2.imread(filename)
        img_bytes = cv2.imencode('.jpg', img)[1]
        a = ImageIndex(img_bytes)
        assert a.frame_colorlayer_detect()['blue'] == '35.04%' and \
               a.frame_colorlayer_detect()['green'] == '30.59%'

    # 暂时还无接口
    def error_frame_detection_test(self):
        pass

    def test_green_frame(self):
        filename = self.module_path + '/image_data/green.png'
        img = cv2.imread(filename)
        img_bytes = cv2.imencode('.png', img)[1]
        a = ImageIndex(img_bytes)
        assert a.green_frame_detect() is not None


if __name__ == '__main__':
    pytest.main(["-s", "func_test.py"])
