""" 图像质量检测库
"""
import json
import time

import cv2
import numpy
import requests
from PIL import Image
from cnocr import CnOcr
from cnstd import CnStd
from skimage.measure import compare_ssim


class ImageSplitJoint(object):
    """图像帧拼接识别类
    """

    def __init__(self, img: numpy.ndarray = None):
        """
        :param img: 输入图像
        """
        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result_line_list = []

    def __get_img_info(self):
        """获取图像的长宽信息
        :return: (height, width)
        """
        img_shape = self.img.shape
        return img_shape[0], img_shape[1]

    def line_detect(self):
        """ 拼接线检测
        :return:
        """
        gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        # 边缘检测
        canny = cv2.Canny(gray_img, 25, 50)
        # 获取宽
        width = gray_img.shape[1]
        # 直线检测
        lines = cv2.HoughLinesP(canny, 1, numpy.pi / 180,
                                threshold=int(width * 0.5), minLineLength=width * 0.8,
                                maxLineGap=width * 0.1)
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    if y1 == y2:
                        self.result_line_list.append(((x1, y1), (x2, y2)))
                    else:
                        continue
        else:
            pass

        if len(self.result_line_list) > 0:
            return True
        else:
            return False


class ImageClarity(object):
    """
    图像清晰度类
    """

    def __init__(self, img: numpy.ndarray = None):
        """
        :param img: 输入图像
        """
        self.image_data = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度图
        self.image_data = cv2.resize(self.image_data, (1920, 1080), )  # 将图像resize成特定大小再判断

    @staticmethod
    def __sobel(image):
        """
        使用Sobel算子提取水平和竖直方向上的边缘信息，并返回其对应的梯度图像
        """
        x = cv2.Sobel(image, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(image, cv2.CV_16S, 0, 1)
        absX = cv2.convertScaleAbs(x)  # 转回uint8
        absY = cv2.convertScaleAbs(y)
        dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        return dst

    @staticmethod
    def __getBlock(G, Gr, block_size):
        """
        找出图像信息最丰富的多个块，并计算其结构清晰度NRSS
        """
        (h, w) = G.shape
        G_blk_list = []
        Gr_blk_list = []
        for i in range(block_size):
            for j in range(block_size):
                G_blk = G[int((i / block_size) * h):int(((i + 1) / block_size) * h),
                        int((j / block_size) * w):int(((j + 1) / block_size) * w)]
                Gr_blk = Gr[int((i / block_size) * h):int(((i + 1) / block_size) * h),
                         int((j / block_size) * w):int(((j + 1) / block_size) * w)]
                G_blk_list.append(G_blk)
                Gr_blk_list.append(Gr_blk)
        sum = 0
        for i in range(block_size * block_size):
            mssim = compare_ssim(G_blk_list[i], Gr_blk_list[i])
            sum += mssim
        nrss = 1 - sum / (block_size * block_size * 1.0)
        return nrss

    def get_frame_clarity_laplacian(self):
        """清晰度检测（拉普拉斯）
        :return:进行拉普拉斯算法之后的方差
        """
        imageVar = cv2.Laplacian(self.image_data, cv2.CV_64F).var()
        return imageVar

    def get_frame_clarity_nrss(self):
        """清晰度检测（NRSS）
        :return:
        """
        gblur_image = cv2.GaussianBlur(self.image_data, (7, 7), 0)  # 高斯滤波
        G = self.__sobel(self.image_data)
        Gr = self.__sobel(gblur_image)
        block_size = 6
        return self.__getBlock(G, Gr, block_size)  # 获取块信息


class GreenImage(object):
    """
    绿屏类
    """

    def __init__(self, img: numpy.ndarray = None):
        """
        :param img: 输入图像
        """
        self.image_data = Image.fromarray(img)
        self.image_data = self.image_data.convert('HSV')
        self.width, self.height = self.image_data.size

    def get_green_frame(self):
        clrs = self.image_data.getcolors()  # 默认至多得到128种像素，检测超过会返回none
        if clrs:
            green_list = [0] * 65
            greens = 0  # 像素的色相为绿的总个数
            for clr in clrs:
                # 绿色的H色相值为35～99（绿+青）
                if 35 <= clr[1][0] <= 99:
                    green_list[clr[1][0] - 35] += clr[0]
                    greens += clr[0]
            details = []
            for i in range(65):
                if green_list[i] > 0:
                    detail = {"H_value": i + 35, "count": green_list[i]}
                    details.append(detail)
            res = {"green_ratio": greens / (self.width * self.height), "details": details}
            return res
        else:
            return None


class ImageColorLayer(object):
    def __init__(self, img: numpy.ndarray = None):
        m, n, c = img.shape
        size = (int(n * 0.5), int(m * 0.5))
        self.img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)  # 降低运算量
        self.hsv = cv2.cvtColor(self.img, cv2.COLOR_RGB2HSV)
        self.red_lower1 = numpy.array([0, 43, 46])
        self.red_upper1 = numpy.array([10, 255, 255])
        self.red_lower2 = numpy.array([153, 43, 46])
        self.red_upper2 = numpy.array([180, 255, 255])
        self.green_lower = numpy.array([35, 43, 46])
        self.green_upper = numpy.array([99, 255, 255])
        self.blue_lower = numpy.array([100, 43, 46])
        self.blue_upper = numpy.array([130, 255, 255])
        self.color_list = [0, 1, 2]
        self.color_map = numpy.zeros((m, n), numpy.int8)
        self.color_dict = {'blue': '', 'green': '', 'red': '', 'exist_deep_red': False}

    def get_colorlayer_info(self):
        self.__get_colors_ratio()
        self.__get_red_ratio()
        return self.color_dict

    def __get_red_ratio(self):
        """计算具有固定数值的图层颜色所占比例，针对深红色用于判断有无
        """
        deep_red_area = 0
        m, n, c = self.img.shape
        for i in range(m):
            for j in range(n):
                pixel = self.img[i, j]
                b, g, r = pixel[0], pixel[1], pixel[2]  # 图片在传输时变为rgb
                if 118 <= r <= 138 and 118 <= g <= 138 and 245 <= b <= 255:
                    deep_red_area += 1  # 深红图层
        deep_red_ratio = deep_red_area / (m * n)
        deep_red_flag = False
        if deep_red_ratio < 0.001:   # 深红占比少于0.1%认为不存在
            pass
        else:
            deep_red_flag = True
        self.color_dict['exist_deep_red'] = deep_red_flag

    def __get_colors_ratio(self):
        """计算红绿蓝图层的比例
        """
        for color_index in self.color_list:
            if color_index == 0:  # 红色
                red_area = 0
                mask1 = cv2.inRange(self.hsv, self.red_lower1, self.red_upper1)
                mask2 = cv2.inRange(self.hsv, self.red_lower2, self.red_upper2)
                mask = cv2.bitwise_or(mask1, mask2)
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                m, n = mask.shape
                for cnt in contours:
                    cv2.fillPoly(mask, [cnt], (255, 255, 255))
                for i in range(m):
                    for j in range(n):
                        if mask[i][j] != 0:
                            red_area += 1
                            self.color_map[i][j] = 1  # 做标记防止重复计算
                red_ratio = '{:.2%}'.format(red_area / (m * n))
                self.color_dict['red'] = red_ratio
            elif color_index == 1:  # 绿色
                green_area = 0
                mask = cv2.inRange(self.hsv, self.green_lower, self.green_upper)
                cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                m, n = mask.shape
                for i in range(m):
                    for j in range(n):
                        if mask[i][j] != 0 and self.color_map[i][j] == 0:
                            green_area += 1
                            self.color_map[i][j] = 2
                red_ratio = '{:.2%}'.format(green_area / (m * n))
                self.color_dict['green'] = red_ratio
            elif color_index == 2:  # 蓝色
                blue_area = 0
                mask = cv2.inRange(self.hsv, self.blue_lower, self.blue_upper)
                cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                m, n = mask.shape
                for i in range(m):
                    for j in range(n):
                        if mask[i][j] != 0 and self.color_map[i][j] == 0:
                            blue_area += 1
                red_ratio = '{:.2%}'.format(blue_area / (m * n))
                self.color_dict['blue'] = red_ratio


class ImageQualityIndexGenerator(object):
    """ 图像质量指标库
    """

    def __init__(self, image_file):
        """
        """
        self.__blurred_frame_check_server_url = "http://172.22.119.82:8601/v1/models/blurred_screen_model:predict"
        self.image_data = self.__bytesIO2img(image_file)

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

    def get_horizontal_portrait_frame_size(self):
        """ 判断横竖屏，并获取横竖屏比例
        :return:
        """
        image_split_joint_handler = ImageSplitJoint(self.image_data)
        return image_split_joint_handler.line_detect()

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

    def get_image_clarity(self):
        image_clarity_handler = ImageClarity(self.image_data)
        return image_clarity_handler.get_frame_clarity_nrss()

    def get_image_colorlayer(self):
        image_colorlayer_handler = ImageColorLayer(self.image_data)
        return image_colorlayer_handler.get_colorlayer_info()


if __name__ == '__main__':
    pass
