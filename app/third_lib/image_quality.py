""" 图像质量检测库
"""
import cv2
import json
import io
import numpy
import requests
from cnocr import CnOcr
from cnstd import CnStd
from PIL import Image
import time


class ImageQualityIndexGenerator(object):
    """ 图像质量指标库
    """

    def __init__(self, image_file):
        """
        """
        self.__blurred_frame_check_server_url = "http://172.22.119.82:8601/v1/models/blurred_screen_model:predict"
        self.image_data = self.__bytesIO2img(image_file)
        # self.__img_std = CnStd()
        # self.__img_ocr = CnOcr()

    def __bytesIO2img(self, image_file):
        """
        :return:
        """
        # in_memory_file = io.BytesIO()
        # image_file.save(in_memory_file)
        img = numpy.frombuffer(image_file, dtype=numpy.uint8)
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
                print(err)
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
        __img_std = CnStd()
        __img_ocr = CnOcr(name=str(now))
        image = Image.fromarray(self.image_data)  # 先转格式为Image 为了统一输入图像尺寸
        if (image.size[0] * image.size[1]) > 20000:
            predict_image = image.resize(
                (int(image.size[0] * 0.5), int(image.size[1] * 0.5)),
                Image.NEAREST
            )
        else:
            predict_image = image
        img = numpy.asarray(predict_image)
        box_info_list = __img_std.detect(img, pse_min_area=500)
        print(time.time() - now)
        ocr_result_list = []
        for box_info in box_info_list:
            cropped_img = box_info['cropped_img']
            ocr_res = __img_ocr.ocr_for_single_line(cropped_img)
            ocr_result_list.append(''.join(ocr_res))
        print(time.time() - now)
        del __img_std
        del __img_ocr
        return ocr_result_list

    def get_if_blurred_frame(self):
        """ 花屏检测
        :return:
        """
        img = cv2.cvtColor(self.image_data, cv2.COLOR_RGB2GRAY)
        img_laplace = self.__laplace_image(img)
        img = cv2.cvtColor(img_laplace, cv2.COLOR_GRAY2RGB)
        image = Image.fromarray(img)  # 先转格式为Image 为了统一输入图像尺寸
        predict_image = image.resize((960, 448), Image.NEAREST)
        img = numpy.asarray(predict_image).astype(float)
        img_list = img.tolist()
        request_url = self.__blurred_frame_check_server_url
        predict_result = self.__access_model_server(img_list, request_url)
        # print(predict_result)
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
