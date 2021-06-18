import os
import time

import cv2

import numpy as np
import matplotlib.pyplot as plt
from PIL.Image import Image
from cnocr import CnOcr
from cnstd import CnStd

from scipy.ndimage import variance
from skimage import io
from skimage.color import rgb2gray
from skimage.filters import laplace
from skimage.transform import resize


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

    @staticmethod
    def cut__avg_frame(file_name):
        """
        :return:
        """
        img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        hight, width = img.shape
        print(hight, width)
        cropped1 = img[0:hight//2, 0:width//2]
        cropped2 = img[hight//2:hight, 0:width//2]
        cropped3 = img[0:hight//2, width//2:width]
        cropped4 = img[hight//2:hight, width//2:width]
        cv2.imwrite('cropped1.png', cropped1)
        cv2.imwrite('cropped2.png', cropped2)
        cv2.imwrite('cropped3.png', cropped3)
        cv2.imwrite('cropped4.png', cropped4)

    def img_var(self, dir_path1, dir_path2=None):
        """
        :return:
        """
        blur_image_list_x = []
        blur_image_list_y = []

        path2_image_list_x = []
        path2_image_list_y = []
        for file_name in os.listdir(dir_path1):
            file_path = os.path.join(dir_path1, file_name)
            img = io.imread(file_path)
            # img = resize(img, (800, 500))
            img = rgb2gray(img)
            edge_laplace = laplace(img, ksize=6)
            blur_image_list_x.append(variance(edge_laplace))
            blur_image_list_y.append(np.amax(edge_laplace))
            print(f"Variance: {variance(edge_laplace)}", f"Maximum : {np.amax(edge_laplace)}", file_path)

        for file_name in os.listdir(dir_path2):
            file_path = os.path.join(dir_path2, file_name)
            img = io.imread(file_path)
            # img = resize(img, (800, 500))
            img = rgb2gray(img)
            edge_laplace = laplace(img, ksize=6)

            path2_image_list_x.append(variance(edge_laplace))
            path2_image_list_y.append(np.amax(edge_laplace))
            print(f"Variance: {variance(edge_laplace)}", f"Maximum : {np.amax(edge_laplace)}", file_path)

        x = np.array(blur_image_list_x)
        y = np.array(blur_image_list_y)
        plt.scatter(x, y)

        plt.scatter(path2_image_list_x, path2_image_list_y)

        plt.show()

    @staticmethod
    def alpha_detect(image_path):
        """
        """
        img = cv2.imread(image_path)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        cv2.imshow("hsv", img_hsv)
        mask1 = cv2.inRange(img_hsv, (0, 50, 20), (5, 255, 255))
        mask2 = cv2.inRange(img_hsv, (175, 50, 20), (180, 255, 255))

        mask = cv2.bitwise_or(mask1, mask2)
        croped = cv2.bitwise_and(img, img, mask=mask)

        cv2.imshow("mask", mask)
        cv2.imshow("croped", croped)
        cv2.waitKey()

    @staticmethod
    def get_ocr_result_list(image_path):
        """ 图像ocr
        :return:
        """
        now = time.time()
        __img_std = CnStd()
        __img_ocr = CnOcr(name=str(now))

        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
        """
        image = Image.fromarray(image_data)  # 先转格式为Image 为了统一输入图像尺寸
        if (image.size[0] * image.size[1]) > 20000:
            predict_image = image.resize(
                (int(image.size[0] * 0.5), int(image.size[1] * 0.5)),
                Image.NEAREST
            )
        else:
            predict_image = image
        img = np.asarray(predict_image)
        """
        box_info_list = __img_std.detect(image_path, pse_min_area=500)
        print(time.time() - now)
        ocr_result_list = []
        for box_info in box_info_list:
            cropped_img = box_info['cropped_img']
            ocr_res = __img_ocr.ocr_for_single_line(cropped_img)
            ocr_result_list.append(''.join(ocr_res))
        print(ocr_result_list)
        print(time.time() - now)
        del __img_std
        del __img_ocr
        return ocr_result_list

    @staticmethod
    def ocr_test(image_path):
        """
        """
        t1 = time.time()
        std = CnStd()
        cn_ocr = CnOcr()

        box_info_list = std.detect(image_path, max_size=1000, pse_min_area=500)
        print(time.time() - t1)
        for box_info in box_info_list:
            # print(box_info)
            cropped_img = box_info['cropped_img']  # 检测出的文本框
            box_list = np.mean(box_info['box'], axis=0)
            # print(box_list)
            # break
            cropped_img = cv2.flip(cropped_img, -1)
            # cv2.imwrite('res.png', cropped_img)
            ocr_res = cn_ocr.ocr_for_single_line(cropped_img)
            print('ocr result: %s' % ''.join(ocr_res))
        print(time.time() - t1)


if __name__ == '__main__':
    BlurredImageChecker().ocr_test('/Users/luoyadong/Desktop/1.png')
