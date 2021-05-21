"""
重要类函数单元测试
"""

import os

import cv2
import pytest

from app.model import ImageIndex, PlayerIndex


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
        filename = self.module_path + '/image_data/color_layer_detect2.png'
        img = cv2.imread(filename)
        img_bytes = cv2.imencode('.png', img)[1]
        a = ImageIndex(img_bytes)
        assert a.frame_colorlayer_detect()['blue'] == '37.33%' and \
               a.frame_colorlayer_detect()['green'] == '26.80%'

    def test_horizontal_frame_detect_true(self):
        filename = self.module_path + '/image_data/horizontal_frame_detect_true.png'
        img = cv2.imread(filename)
        img_bytes = cv2.imencode('.png', img)[1]
        a = ImageIndex(img_bytes)
        assert a.frame_horizontal_portrait_detect() is True

    def test_horizontal_frame_detect_false(self):
        filename = self.module_path + '/image_data/horizontal_frame_detect_false.png'
        img = cv2.imread(filename)
        img_bytes = cv2.imencode('.png', img)[1]
        a = ImageIndex(img_bytes)
        assert a.frame_horizontal_portrait_detect() is False

    def test_get_colour_cast_true(self):
        filepath = self.module_path + '/video_data/get_colour_cast_true.mov'
        a = PlayerIndex(colour_cast_dict={"video_path": filepath})
        assert a.get_colour_cast_index()['judge'] is True

    def test_get_colour_cast_false(self):
        filepath = self.module_path + '/video_data/get_colour_cast_false.mp4'
        a = PlayerIndex(colour_cast_dict={"video_path": filepath})
        assert a.get_colour_cast_index()['judge'] is False

    def test_get_colour_cast_with_reference(self):
        file_src_path = self.module_path + '/video_data/get_colour_cast_false.mp4'
        file_target_path = self.module_path + '/video_data/get_colour_cast_false.mp4'
        a = PlayerIndex(colour_cast_dict={"src_video_path": file_src_path, "target_video_path": file_target_path})
        assert a.get_colour_cast_index_with_reference()['judge'] is False

    # 暂时还无接口
    def error_frame_detection_test(self):
        pass

    def test_green_frame(self):
        filename = self.module_path + '/image_data/green.png'
        img = cv2.imread(filename)
        img_bytes = cv2.imencode('.png', img)[1]
        a = ImageIndex(img_bytes)
        assert a.green_frame_detect() is not None

    def test_get_video_quality_vmaf(self):
        input_video_path = os.path.join(self.module_path, "video_data/vmaf_refer_video.mp4")
        refer_video_path = os.path.join(self.module_path, "video_data/vmaf_input_video.mp4")
        a = PlayerIndex(video_quality_dict={"input_video": input_video_path, "refer_video": refer_video_path})
        assert a.get_video_quality_vmaf()['vmaf_score'] is not None


if __name__ == '__main__':
    pytest.main(["-s", "func_test.py"])
