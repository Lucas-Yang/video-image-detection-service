""" 图像质量检测库
"""
import cv2
import json
import io
import numpy
import requests
from cnocr import CnOcr
from cnstd import CnStd
import time


class ImageQualityIndexGenerator(object):
    """ 图像质量指标库
    """

    def __init__(self, image_file):
        """
        """
        self.__blurred_frame_check_server_url = ""
        self.image_data = self.__bytesIO2img(image_file)
        self.__img_std = CnStd()
        self.__img_ocr = CnOcr()

    def __bytesIO2img(self, image_file):
        """
        :return:
        """
        in_memory_file = io.BytesIO()
        image_file.save(in_memory_file)
        img = numpy.fromstring(in_memory_file.getvalue(), numpy.uint8)
        img = cv2.imdecode(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __access_model_server(self, img, request_url):

        """访问接口
            :param img:
            :param request_url:
            :return:
        """
        headers = {"content-type": "application/json"}
        body = {"instances": [{"input_1": img}]}
        try_times = 0
        while try_times < 3:
            try:
                response = requests.post(request_url, data=json.dumps(body), headers=headers)
                response.raise_for_status()
                prediction = response.json()['predictions'][0]
                return numpy.argmax(prediction)
            except Exception as err:
                try_times += 1
                continue
        if try_times >= 3:
            return -1

    def __image_ssim_sim(self):
        """
        :return:
        """
        pass

    def __image_psnr_sim(self):
        """
        :return:
        """
        pass

    def __image_hash_sim(self):
        """
        :return:
        """
        pass

    @staticmethod
    def __laplace_image(image_np):
        """ 花屏训练数据处理，拉普拉斯变换
        :return:
        """
        img_laplace = cv2.Laplacian(image_np, cv2.CV_16S, ksize=3)
        img_laplace = cv2.convertScaleAbs(img_laplace)
        return img_laplace

    def get_black_white_frame_score(self):
        """ 判断是否是黑屏，白屏，蓝屏等
        :return: true / false
        """
        gaussian_image = cv2.GaussianBlur(self.image_data,
                                          ksize=(9, 9),
                                          sigmaX=0,
                                          sigmaY=0)
        return gaussian_image.var()

    def get_ocr_result_list(self):
        """ 图像ocr
        :return:
        """
        now = time.time()
        box_info_list = self.__img_std.detect(self.image_data, pse_min_area=500)
        print(time.time() - now)
        ocr_result_list = []
        for box_info in box_info_list:
            cropped_img = box_info['cropped_img']
            ocr_res = self.__img_ocr.ocr_for_single_line(cropped_img)
            ocr_result_list.append(''.join(ocr_res))
        print(time.time() - now)
        return ocr_result_list

    def get_if_blurred_frame(self):
        """ 花屏检测
        :return:
        """
        img_gray = cv2.cvtColor(self.image_data, cv2.COLOR_RGB2GRAY)
        img_laplace = self.__laplace_image(img_gray)
        img = cv2.cvtColor(img_laplace, cv2.COLOR_GRAY2RGB)
        print(cv2.cvtColor(img_laplace, cv2.COLOR_GRAY2RGB).shape)
        print(img_laplace.shape)
        # return True
        request_url = self.__blurred_frame_check_server_url
        predict_result = self.__access_model_server(img, request_url)
        if predict_result == -1:
            return -1
        else:
            return predict_result

    def get_image_ssim(self):
        """ 图像结构相似度相似度
        :return:
        """
        pass


if __name__ == '__main__':
    pass
