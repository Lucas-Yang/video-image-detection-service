"""
接口单元测试
"""
import os
import time

import pytest
import requests


class TestPort(object):
    module_path = os.path.dirname(__file__)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) "
                             "Chrome/51.0.2704.103 Safari/537.36"}

    def test_green_frame_detect(self):
        url = "http://127.0.0.1:8090/image/quality/green-frame-detect"
        filepath = self.module_path + '/image_data/green.png'
        files = {'file': open(filepath, "rb")}
        response = requests.post(url=url, files=files, headers=self.headers)
        assert response.status_code == 200
        assert response.json()['code'] == 0

    def test_clarity_detect(self):
        url = "http://127.0.0.1:8090/image/quality/clarity-detect"
        filepath = self.module_path + '/image_data/clarity.jpeg'
        files = {'file': open(filepath, "rb")}
        response = requests.post(url=url, files=files, headers=self.headers)
        assert response.status_code == 200
        # print(response.json())
        assert response.json()['code'] == 0

    def test_get_silence_index(self):
        url = "http://127.0.0.1:8090/player/index/silence"
        filepath = self.module_path + '/image_data/silence.mp3'
        files = {'file_src': open(filepath, "rb")}
        response = requests.post(url=url, files=files, headers=self.headers, )
        assert response.status_code == 200
        # print(response.json())
        assert response.json()['code'] == 0

    def test_judge_black_white_frame(self):
        url = "http://localhost:8090/image/quality/black_white-detect"
        filepath = self.module_path + '/image_data/white.jpg'
        files = [('file', ('white.jpg', open(filepath, 'rb'), 'image/png'))]
        response = requests.post(url=url, files=files, headers=self.headers, )
        assert response.status_code == 200
        assert response.json()['code'] == 0

    def test_horizontal_frame_detect(self):
        url = "http://127.0.0.1:8090/image/quality/horizontal-frame-detect"
        filepath = self.module_path + '/image_data/black.png'
        files = [('file', ('black.png', open(filepath, 'rb'), 'image/png'))]
        response = requests.post(url=url, files=files, headers=self.headers, )
        assert response.status_code == 200
        assert response.json()['code'] == 0

    def test_blurred_frame_detect(self):
        url = "http://127.0.0.1:8090/image/quality/blurred-detect"
        filepath = self.module_path + '/image_data/cate1.jpg'
        files = [('file', ('white.jpg', open(filepath, 'rb'), 'image/png'))]
        response = requests.post(url=url, files=files, headers=self.headers, )
        assert response.status_code == 200
        assert response.json()['code'] == 0

    def test_color_layer(self):
        url = "http://localhost:8090/image/quality/colorlayer-detect"
        filepath = self.module_path + '/image_data/color_layer_detect2.png'
        files = [('file', ('color_layer_detect2.png', open(filepath, 'rb'), 'image/png'))]
        response = requests.request("POST", url, headers=self.headers, files=files)
        assert response.text.find('"blue":"37.33%","green":"26.80%"') != -1

    def test_watermark_detect(self):
        url = "http://localhost:8090/image/quality/watermark-detect"
        filepath = self.module_path + '/image_data/douyin1.png'
        files = [('file', ('douyin1.png', open(filepath, 'rb'), 'image/png'))]
        response = requests.request("POST", url, headers=self.headers, files=files)
        assert response.text.find('抖音') != -1

    def test_horizontal_frame_detect_v1(self):
        url = "http://localhost:8090/image/quality/horizontal-frame-detect"
        filepath = self.module_path + '/image_data/horizontal_frame_detect_true.png'
        files = [('file', ('horizontal_frame_detect_true.png', open(filepath, 'rb'), 'image/png'))]
        response = requests.request("POST", url, headers=self.headers, files=files)
        assert response.text.find('true') != -1

    def test_frame_orc(self):
        url = "http://localhost:8090/image/quality/char-recognize"
        filepath = self.module_path + '/image_data/ocr.jpg'
        files = [('file', ('ocr.jpg', open(filepath, 'rb'), 'image/png'))]
        response = requests.request("POST", url, headers=self.headers, files=files)
        assert response.json()['code'] == 0

    def test_frame_paddle_orc(self):
        url = "http://localhost:8090/image/quality/char-recognize"
        filepath = self.module_path + '/image_data/ocr.jpg'
        files = [('file', ('ocr.jpg', open(filepath, 'rb'), 'image/png'))]
        response = requests.request("POST", url, headers=self.headers, files=files)
        assert response.json()['code'] == 0

    def test_get_colour_cast(self):
        url = "http://localhost:8090/player/video/colorcast-detect"
        filepath = self.module_path + '/video_data/get_colour_cast_true.mov'
        files = [('file_src', ('get_colour_cast_true.mov', open(filepath, 'rb'), 'video/mov'))]
        response = requests.request("POST", url, headers=self.headers, files=files)
        assert response.text.find('true') != -1

    def test_get_colour_cast_with_reference(self):
        url = "http://localhost:8090/player/video/colorcast-detect"
        file_src_path = self.module_path + '/video_data/get_colour_cast_false.mp4'
        file_target_path = self.module_path + '/video_data/get_colour_cast_false.mp4'
        files = [
            ('file_src', ('get_colour_cast_false.mp4', open(file_src_path, 'rb'), 'video/mp4')),
            ('file_target', ('get_colour_cast_false.mp4', open(file_target_path, 'rb'), 'video/mp4'))
        ]
        response = requests.request("POST", url, headers=self.headers, files=files)
        assert response.text.find('false') != -1

    def test_get_vmaf_score(self):
        url = "http://127.0.0.1:8090/player/video/vmaf"
        input_video_path = os.path.join(self.module_path, "video_data/vmaf_input_video.mp4")
        refer_video_path = os.path.join(self.module_path, "video_data/vmaf_refer_video.mp4")
        files = [
            ('file_input', ('vmaf_input_video.mp4', open(input_video_path, 'rb'), 'video/mp4')),
            ('file_refer', ('vmaf_refer_video.mp4', open(refer_video_path, 'rb'), 'video/mp4'))
        ]
        response = requests.post(url=url, files=files, headers=self.headers, )
        assert response.json()['code'] == 0

    def test_get_video_ssim(self):
        url = "http://127.0.0.1:8090/player/video/ssim"
        file_src_path = self.module_path + '/video_data/get_colour_cast_false.mp4'
        file_target_path = self.module_path + '/video_data/get_colour_cast_false.mp4'
        files = [
            ('file_src', ('get_colour_cast_false.mp4', open(file_src_path, 'rb'), 'video/mp4')),
            ('file_target', ('get_colour_cast_false.mp4', open(file_target_path, 'rb'), 'video/mp4'))
        ]
        response = requests.request("POST", url, headers=self.headers, files=files)
        assert response.json()['code'] == 0

    def test_get_image_match_res(self):
        url = "http://localhost:8090/image/quality/image-match"
        file_src_path = self.module_path + '/image_data/horizontal_frame_detect_false.png'
        file_target_path = self.module_path + '/image_data/horizontal_frame_detect_false.png'
        files = [('file_src', ('horizontal_frame_detect_false.png', open(file_src_path, 'rb'), 'image/png')),
                 ('file_target', ('horizontal_frame_detect_false.png', open(file_target_path, 'rb'), 'image/png'))]
        response = requests.request("POST", url, headers=self.headers, files=files)
        assert response.json()['code'] == 0

    def test_get_image_ssim(self):
        url = "http://localhost:8090/image/quality/similarity"
        file_src_path = self.module_path + '/image_data/similarity_src.png'
        file_target_path = self.module_path + '/image_data/similarity_target.png'
        files = [('file_src', ('similarity_src.png', open(file_src_path, 'rb'), 'image/png')),
                 ('file_target', ('similarity_target.png', open(file_target_path, 'rb'), 'image/png'))]
        response = requests.request("POST", url, headers=self.headers, files=files)
        assert response.json()['code'] == 0

    def test_get_image_ssim_v2(self):
        url = "http://localhost:8090/image/quality/similarity-v2"
        file_src_path = self.module_path + '/image_data/similarity_src.png'
        file_target_path = self.module_path + '/image_data/similarity_target.png'
        files = [('file_src', ('similarity_src.png', open(file_src_path, 'rb'), 'image/png')),
                 ('file_target', ('similarity_target.png', open(file_target_path, 'rb'), 'image/png'))]
        response = requests.request("POST", url, headers=self.headers, files=files)
        assert response.json()['code'] == 0

    def test_get_video_black(self):
        upload_url = "http://localhost:8090/player/video/upload"
        file_path = self.module_path +"/video_data/black.mp4"
        files = [('file', ('black.mp4', open(file_path, 'rb'), 'video/mp4'))]
        payload = {'index_types': 'BLACKFRAME',
                   'black_threshold': '0.9'}
        response = requests.request("POST", upload_url, headers=self.headers, data=payload, files=files)
        assert response.json()['code'] == 0
        ctr = 0
        while True:
            ctr += 1
            cv_index_url = "http://localhost:8090/player/index/cv?task_id=" + response.json()['task_id']
            result = requests.request("GET", cv_index_url, headers=self.headers)
            if result.json()["code"] != -4:
                assert result.json()["data"]["black_frame_list"][0]['black_duration'] == '59.96'
                return
            if ctr > 120:
                raise TimeoutError("Time out")
            time.sleep(10)

    def test_get_video_startapp(self):
        upload_url = "http://localhost:8090/player/video/upload"
        file_path = self.module_path + "/video_data/startapp.mp4"
        files = [('file', ('startapp.mp4', open(file_path, 'rb'), 'video/mp4'))]
        payload = {'index_types': 'STARTAPP'}
        response = requests.request("POST", upload_url, headers=self.headers, data=payload, files=files)
        assert response.json()['code'] == 0
        ctr = 0
        while True:
            ctr += 1
            cv_index_url = "http://localhost:8090/player/index/cv?task_id=" + response.json()['task_id']
            result = requests.request("GET", cv_index_url, headers=self.headers)
            if result.json()["code"] != -4:
                assert result.json()["data"]["start_app_time"] == 1.3125
                return
            if ctr > 120:
                raise TimeoutError("Time out")
            time.sleep(10)

    def test_get_video_startappyouku(self):
        upload_url = "http://localhost:8090/player/video/upload"
        file_path = self.module_path + "/video_data/startyouku.mp4"
        files = [('file', ('startyouku.mp4', open(file_path, 'rb'), 'video/mp4'))]
        payload = {'index_types': 'STARTAPPYOUKU'}
        response = requests.request("POST", upload_url, headers=self.headers, data=payload, files=files)
        assert response.json()['code'] == 0
        ctr = 0
        while True:
            ctr += 1
            cv_index_url = "http://localhost:8090/player/index/cv?task_id=" + response.json()['task_id']
            result = requests.request("GET", cv_index_url, headers=self.headers)
            if result.json()["code"] != -4:
                assert result.json()["data"]["start_app_time"] == 5.90625
                return
            if ctr > 120:
                raise TimeoutError("Time out")
            time.sleep(10)
    
    def test_get_video_startappxigua(self):
        upload_url = "http://localhost:8090/player/video/upload"
        file_path = self.module_path + "/video_data/startxigua.mp4"
        files = [('file', ('startxigua.mp4', open(file_path, 'rb'), 'video/mp4'))]
        payload = {'index_types': 'STARTAPPIXIGUA'}
        response = requests.request("POST", upload_url, headers=self.headers, data=payload, files=files)
        assert response.json()['code'] == 0
        ctr = 0
        while True:
            ctr += 1
            cv_index_url = "http://localhost:8090/player/index/cv?task_id=" + response.json()['task_id']
            result = requests.request("GET", cv_index_url, headers=self.headers)
            if result.json()["code"] != -4:
                assert result.json()["data"]["start_app_time"] == 2.15625
                return
            if ctr > 120:
                raise TimeoutError("Time out")
            time.sleep(10)
    
    def test_get_video_startapptencent(self):
        upload_url = "http://localhost:8090/player/video/upload"
        file_path = self.module_path + "/video_data/starttencent.mp4"
        files = [('file', ('starttencent.mp4', open(file_path, 'rb'), 'video/mp4'))]
        payload = {'index_types': 'STARTAPPTENCENT'}
        response = requests.request("POST", upload_url, headers=self.headers, data=payload, files=files)
        assert response.json()['code'] == 0
        ctr = 0
        while True:
            ctr += 1
            cv_index_url = "http://localhost:8090/player/index/cv?task_id=" + response.json()['task_id']
            result = requests.request("GET", cv_index_url, headers=self.headers)
            if result.json()["code"] != -4:
                assert result.json()["data"]["start_app_time"] == 6.09375
                return
            if ctr > 120:
                raise TimeoutError("Time out")
            time.sleep(10)
    
    def test_get_video_startappiqiyi(self):
        upload_url = "http://localhost:8090/player/video/upload"
        file_path = self.module_path + "/video_data/startiqiyi.mp4"
        files = [('file', ('startiqiyi.mp4', open(file_path, 'rb'), 'video/mp4'))]
        payload = {'index_types': 'STARTAPPIQIYI'}
        response = requests.request("POST", upload_url, headers=self.headers, data=payload, files=files)
        assert response.json()['code'] == 0
        ctr = 0
        while True:
            ctr += 1
            cv_index_url = "http://localhost:8090/player/index/cv?task_id=" + response.json()['task_id']
            result = requests.request("GET", cv_index_url, headers=self.headers)
            if result.json()["code"] != -4:
                assert result.json()["data"]["start_app_time"] == 5.53125
                return
            if ctr > 120:
                raise TimeoutError("Time out")
            time.sleep(10)
    
    def test_get_video_startappdouyin(self):
        upload_url = "http://localhost:8090/player/video/upload"
        file_path = self.module_path + "/video_data/startdouyin.mp4"
        files = [('file', ('startdouyin.mp4', open(file_path, 'rb'), 'video/mp4'))]
        payload = {'index_types': 'STARTAPPDOUYIN'}
        response = requests.request("POST", upload_url, headers=self.headers, data=payload, files=files)
        assert response.json()['code'] == 0
        ctr = 0
        while True:
            ctr += 1
            cv_index_url = "http://localhost:8090/player/index/cv?task_id=" + response.json()['task_id']
            result = requests.request("GET", cv_index_url, headers=self.headers)
            if result.json()["code"] != -4:
                assert result.json()["data"]["start_app_time"] == 4.3125
                return
            if ctr > 120:
                raise TimeoutError("Time out")
            time.sleep(10)
    
    def test_get_video_freezeframe(self):
        upload_url = "http://localhost:8090/player/video/upload"
        file_path = self.module_path + "/video_data/freezeframe.mp4"
        files = [('file', ('freezeframe.mp4', open(file_path, 'rb'), 'video/mp4'))]
        payload = {'index_types': 'FREEZEFRAME'}
        response = requests.request("POST", upload_url, headers=self.headers, data=payload, files=files)
        assert response.json()['code'] == 0
        ctr = 0
        while True:
            ctr += 1
            cv_index_url = "http://localhost:8090/player/index/cv?task_id=" + response.json()['task_id']
            result = requests.request("GET", cv_index_url, headers=self.headers)
            if result.json()["code"] != -4:
                assert result.json()["data"]['freeze_frame_list'][0]['freeze_start_time'] == '3.84'
                return
            if ctr > 120:
                raise TimeoutError("Time out")
            time.sleep(10)
    
    def test_get_video_firstframe(self):
        upload_url = "http://localhost:8090/player/video/upload"
        file_path = self.module_path + "/video_data/firstframe.mp4"
        files = [('file', ('firstframe.mp4', open(file_path, 'rb'), 'video/mp4'))]
        payload = {'index_types': 'FIRSTFRAME'}
        response = requests.request("POST", upload_url, headers=self.headers, data=payload, files=files)
        assert response.json()['code'] == 0
        ctr = 0
        while True:
            ctr += 1
            cv_index_url = "http://localhost:8090/player/index/cv?task_id=" + response.json()['task_id']
            result = requests.request("GET", cv_index_url, headers=self.headers)
            if result.json()["code"] != -4:
                assert result.json()["data"]["first_frame_time"] == -1
                return
            if ctr > 120:
                raise TimeoutError("Time out")
            time.sleep(10)


if __name__ == '__main__':
    pytest.main(["-s", "port_test.py"])