""" 图像质量检测库
"""
import cv2
import json
import io
import numpy
import requests


class ImageQualityIndexGenerator(object):
    """ 图像质量指标库
    """

    def __init__(self):
        """
        """
        self.__blurred_frame_check_server_url = ""

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
        # img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        img_laplace = cv2.Laplacian(image_np, cv2.CV_16S, ksize=3)
        img_laplace = cv2.convertScaleAbs(img_laplace)
        return img_laplace
        # cv2.imwrite(os.path.join(laplac_dir, file_name + ".jpg"), img_laplace)

    @staticmethod
    def get_black_white_frame_score(image_file):
        """ 判断是否是黑屏，白屏，蓝屏等
        :return: true / false
        """
        in_memory_file = io.BytesIO()
        image_file.save(in_memory_file)
        img = numpy.fromstring(in_memory_file.getvalue(), numpy.uint8)
        img = cv2.imdecode(img, 1)
        gaussian_image = cv2.GaussianBlur(img, ksize=(9, 9), sigmaX=0, sigmaY=0)
        return gaussian_image.var()

    def get_if_blurred_frame(self, image_file):
        """ 花屏检测
        :return:
        """
        in_memory_file = io.BytesIO()
        image_file.save(in_memory_file)
        img = numpy.fromstring(in_memory_file.getvalue(), numpy.uint8)
        img = cv2.imdecode(img, 1)
        print(img.shape)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_laplace = self.__laplace_image(img_gray)
        # cv2.imwrite('test1.jpg', img_laplace)
        print(cv2.cvtColor(img_laplace, cv2.COLOR_GRAY2RGB).shape)
        print(img_laplace.shape)
        return True
        request_url = self.__blurred_frame_check_server_url
        headers = {"content-type": "application/json"}
        body = {"instances": [{"input_1": img}]}
        response = requests.post(request_url, data=json.dumps(body), headers=headers)
        response.raise_for_status()
        prediction = response.json()['predictions'][0]
        # return numpy.argmax(prediction)

    def get_image_ssim(self):
        """ 图像结构相似度相似度
        :return:
        """
        pass


if __name__ == '__main__':
    pass
