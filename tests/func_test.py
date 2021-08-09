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
        assert round(a.black_white_frame_detect()["white_ratio"], 2) == 1

    def test_black_white_frame_detection_false(self):
        filename = self.module_path + '/image_data/black.png'
        img = cv2.imread(filename)
        img_bytes = cv2.imencode('.png', img)[1]
        a = ImageIndex(img_bytes)
        assert round(a.black_white_frame_detect()["black_ratio"], 2) == 1

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
        assert a.blurred_frame_detection() is True

    def test_frame_paddle_ocr(self):
        filename = self.module_path + '/image_data/ocr.jpg'
        img = cv2.imread(filename)
        img_bytes = cv2.imencode('.jpg', img)[1]
        a = ImageIndex(img_bytes)
        assert a.frame_ocr() is not None

    def test_clarity_detect(self):
        filename = self.module_path + '/image_data/clarity.jpeg'
        img = cv2.imread(filename)
        img_bytes = cv2.imencode('.jpeg', img)[1]
        a = ImageIndex(img_bytes)
        assert a.frame_clarity_detect() is not None

    def test_watermark_detect(self):
        filename = self.module_path + '/image_data/douyin1.png'
        img = cv2.imread(filename)
        img_bytes = cv2.imencode('.png', img)[1]
        a = ImageIndex(img_bytes)
        assert a.watermark_frame_detection()[0] == '抖音'

    def test_no_watermark_detect(self):
        filename = self.module_path + '/image_data/kuaishou6.png'
        img = cv2.imread(filename)
        img_bytes = cv2.imencode('.png', img)[1]
        a = ImageIndex(img_bytes)
        assert len(a.watermark_frame_detection()) == 0

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

    def test_get_video_quality_psnr(self):
        input_video_path = os.path.join(self.module_path, "video_data/vmaf_refer_video.mp4")
        refer_video_path = os.path.join(self.module_path, "video_data/vmaf_input_video.mp4")
        a = PlayerIndex(video_quality_dict={"input_video": input_video_path, "refer_video": refer_video_path})
        assert a.get_video_quality_psnr()['psnr_score'] is not None

    def test_get_video_quality_niqe(self):
        input_video_path = os.path.join(self.module_path, "video_data/vmaf_refer_video.mp4")
        a = PlayerIndex(video_quality_dict={"input_video": input_video_path})
        assert a.get_video_quality_niqe()['niqe_score'] is not None

    def test_get_video_quality_brisque(self):
        input_video_path = os.path.join(self.module_path, "video_data/vmaf_refer_video.mp4")
        a = PlayerIndex(video_quality_dict={"input_video": input_video_path})
        assert a.get_video_quality_brisque()['brisque_score'] is not None

    def test_get_video_ssim(self):
        file_src_path = self.module_path + '/video_data/get_colour_cast_false.mp4'
        file_target_path = self.module_path + '/video_data/get_colour_cast_false.mp4'
        a = PlayerIndex(video_quality_dict={"src_video": file_src_path, "target_video": file_target_path})
        assert a.get_video_quality()['ssim_score'] == 1.0

    def test_get_image_match_res(self):
        file_src_path = self.module_path + '/image_data/horizontal_frame_detect_false.png'
        file_target_path = self.module_path + '/image_data/horizontal_frame_detect_false.png'
        res_src = open(file_src_path, 'rb').read()
        target_src = open(file_target_path, 'rb').read()
        a = ImageIndex(quality_file=res_src, target_file=target_src)
        assert a.image_matching()["match_coordinates"][0] == 359

    def test_get_image_ssim(self):
        file_src_path = self.module_path + '/image_data/similarity_src.png'
        file_target_path = self.module_path + '/image_data/similarity_target.png'
        res_src = open(file_src_path, 'rb').read()
        target_src = open(file_target_path, 'rb').read()
        a = ImageIndex(quality_file=res_src, target_file=target_src)
        assert a.calculate_similarity_orb() == 0.03

    def test_get_silence_index(self):
        file_path = self.module_path + '/video_data/silence.mp4'
        a = PlayerIndex(silence_info_dict={"video_path": file_path})
        assert a.get_silence_index()["silence_timestamps"][0]["silence_duration"] == 87.98

    def test_get_video_freezeframe(self):
        file_path = self.module_path + '/video_data/freezeframe.mp4'
        a = PlayerIndex(cv_info_dict={"temp_video_path": file_path, 'index_types': ['FREEZEFRAME']})
        assert a.get_cv_index()['freeze_frame_list'][0]['freeze_start_time'] == '3.84'

    def test_get_video_firstfrmae(self):
        file_path = self.module_path + '/video_data/firstframe.mp4'
        a = PlayerIndex(cv_info_dict={"temp_video_path": file_path, 'index_types': ['FIRSTFRAME']})
        assert a.get_cv_index()["first_frame_time"] == -1

    def test_get_video_startapp(self):
        file_path = self.module_path + '/video_data/startapp.mp4'
        a = PlayerIndex(cv_info_dict={"temp_video_path": file_path, 'index_types': ['STARTAPP']})
        assert a.get_cv_index()["start_app_time"] == 1.3000000000000007

    def test_get_video_startappyouku(self):
        file_path = self.module_path + '/video_data/startyouku.mp4'
        a = PlayerIndex(cv_info_dict={"temp_video_path": file_path, 'index_types': ['STARTAPPYOUKU']})
        assert a.get_cv_index()["start_app_time"] == 5.9

    def test_get_video_startappxigua(self):
        file_path = self.module_path + '/video_data/startxigua.mp4'
        a = PlayerIndex(cv_info_dict={"temp_video_path": file_path, 'index_types': ['STARTAPPIXIGUA']})
        assert a.get_cv_index()["start_app_time"] == 2.0999999999999996

    def test_get_video_startapptencent(self):
        file_path = self.module_path + '/video_data/starttencent.mp4'
        a = PlayerIndex(cv_info_dict={"temp_video_path": file_path, 'index_types': ['STARTAPPTENCENT']})
        assert a.get_cv_index()["start_app_time"] == 6.100000000000001

    def test_get_video_startappiqiyi(self):
        file_path = self.module_path + '/video_data/startiqiyi.mp4'
        a = PlayerIndex(cv_info_dict={"temp_video_path": file_path, 'index_types': ['STARTAPPIQIYI']})
        assert a.get_cv_index()["start_app_time"] == 5.5

    def test_get_video_startappdouyin(self):
        file_path = self.module_path + '/video_data/startdouyin.mp4'
        a = PlayerIndex(cv_info_dict={"temp_video_path": file_path, 'index_types': ['STARTAPPDOUYIN']})
        assert a.get_cv_index()["start_app_time"] == 4.300000000000001


if __name__ == '__main__':
    pytest.main(["-s", "func_test.py"])
