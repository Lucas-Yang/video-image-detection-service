"""
接口单元测试
"""
import os

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
        assert response.status_code is 200

    def test_color_layer(self):
        url = "http://localhost:8090/image/quality/colorlayer-detect"
        filepath = self.module_path + '/image_data/color_layer_detect2.png'
        files = [('file', ('color_layer_detect1.png', open(filepath, 'rb'), 'image/png'))]
        response = requests.request("POST", url, headers=self.headers, files=files)
        assert response.text.find('"blue":"37.33%","green":"26.80%"') != -1

    def test_horizontal_frame_detect(self):
        url = "http://localhost:8090/image/quality/horizontal-frame-detect"
        filepath = self.module_path + '/image_data/horizontal_frame_detect_true.png'
        files = [('file', ('horizontal_frame_detect_true.png', open(filepath, 'rb'), 'image/png'))]
        response = requests.request("POST", url, headers=self.headers, files=files)
        assert response.text.find('true') != -1

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


if __name__ == '__main__':
    pytest.main(["-s", "port_test.py"])
