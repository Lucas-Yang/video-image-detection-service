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
        url = "http://127.0.0.1:8091/image/quality/green-frame-detect"
        filepath = self.module_path + '/image_data/green.png'
        files = {'file': open(filepath, "rb")}
        response = requests.post(url=url, files=files, headers=self.headers)
        assert response.status_code is 200

    def test_color_layer(self):
        url = "http://localhost:8090/image/quality/colorlayer-detect"
        filepath = self.module_path + '/image_data/color_layer_detect1.png'
        files = [('file', ('color_layer_detect1.png', open(filepath, 'rb'), 'image/png'))]
        response = requests.request("POST", url, headers=self.headers, files=files)
        assert response.text.find('"blue":"35.21%","green":"31.55%"') != -1


if __name__ == '__main__':
    pytest.main(["-s", "port_test.py"])
