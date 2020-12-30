import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt


class BlurredImageChecker(object):
    """ 花屏检测test
    """

    def __init__(self, image_path: str = None):
        """
        :param image_path:
        """
        self.image_path = image_path

    def read_image(self, image_name):
        self.image_path = image_name
        img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        # img = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=0, sigmaY=0)
        img_laplac = cv2.Laplacian(img, cv2.CV_16S, ksize=3)
        img_laplac = cv2.convertScaleAbs(img_laplac)
        # img_laplac = cv2.cvtColor(img_laplac, cv2.COLOR_BGR2RGB)
        print(img_laplac.shape)
        plt.imshow(img_laplac, cmap='gray')
        plt.show()
        print(image_name, "方差: ", img_laplac.var())
        return img_laplac.var()

    def gauss_blur(self, image_path):
        """
        :param image_path:
        :return:
        """
        img = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.show()
        # time.sleep(10)
        print("#" * 30)
        gaussian_image = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=0, sigmaY=0)
        # img = cv2.cvtColor(gaussian_image, cv2.COLOR_BGR2RGB)
        plt.imshow(gaussian_image)
        plt.show()

    def read_dir(self, dir_path: str):
        """
        :param dir_path:
        :return:
        """
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            laplac_score = self.read_image(file_path)

    @staticmethod
    def white_frame_detect(image_path):
        """
        :return:
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        gaussian_image = cv2.GaussianBlur(img, ksize=(9, 9), sigmaX=0, sigmaY=0)
        plt.imshow(gaussian_image, cmap='gray')
        plt.show()


if __name__ == '__main__':
    blurred_handler = BlurredImageChecker()
    blurred_handler.white_frame_detect('./1.png')
