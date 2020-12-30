""" 图像质量检测库
"""
import numpy
import cv2
import io


class ImageQualityIndexGenerator(object):
    """ 图像质量指标库
    """

    def __init__(self):
        """
        """
        pass

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

    def get_image_ssim(self):
        """ 图像结构相似度相似度
        :return:
        """
        pass


if __name__ == '__main__':
    pass
