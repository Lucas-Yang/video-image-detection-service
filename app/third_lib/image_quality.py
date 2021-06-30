""" 图像质量检测库
"""
import json
import cv2
import numpy
import random
import requests
from PIL import Image
from cnocr import CnOcr
from cnstd import CnStd
from skimage.measure import compare_ssim


class BlurredFrameDetector(object):
    """ 花屏检测基类
    """
    def __init__(self, image_data):
        self.__blurred_frame_check_server_url = "http://10.221.42.190:8601/v1/models/blurred_screen_model:predict"
        self.image_data = image_data

    @staticmethod
    def access_model_server(img, request_url):
        """ 访问接口
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

    @staticmethod
    def __laplace_image(image_np):
        """ 花屏训练数据处理，拉普拉斯变换
        :return:
        """
        img_laplace = cv2.Laplacian(image_np, cv2.CV_16S, ksize=3)
        img_laplace = cv2.convertScaleAbs(img_laplace)
        return img_laplace

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
        predict_result = self.access_model_server(img_list, request_url)
        if predict_result == -1:
            return -1
        else:
            return predict_result


class WatermarkFrameDetector(object):
    """水印检测类
    """

    def __init__(self, image_data):
        self.__watermark_frame_check_server_url = "http://10.221.42.190:8501/v1/models/watermark_detect_model:predict"
        self.image_data = image_data
        self.image_size = 608  # 性能不行可以调到416
        self.watermark_classes = {
            0: '抖音',
            1: '好看',
            2: '小红书',
            3: '快手',
            4: '快手',
            5: '小红书'
        }
        self.result_list = []

    @staticmethod
    def image_preporcess(image, target_size):
        """图片处理成608*608*3
        """
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(numpy.float32)
        ih, iw = target_size
        h, w, _ = img.shape
        scale = min(iw / w, ih / h)
        nw, nh = int(scale * w), int(scale * h)
        image_resized = cv2.resize(img, (nw, nh))
        image_paded = numpy.full(shape=[ih, iw, 3], fill_value=128.0)
        dw, dh = (iw - nw) // 2, (ih - nh) // 2
        image_paded[dh:nh + dh, dw:nw + dw, :] = image_resized
        image_paded = image_paded / 255.
        return image_paded

    @staticmethod
    def postprocess_boxes(pred_bbox, org_img_shape, input_size, score_threshold):
        """将所有可能的预测信息提取出来，主要是三类：类别，可能性，坐标值
        """
        valid_scale = [0, numpy.inf]
        pred_bbox = numpy.array(pred_bbox)
        pred_xywh = pred_bbox[:, 0:4]
        pred_conf = pred_bbox[:, 4]
        pred_prob = pred_bbox[:, 5:]
        pred_coor = numpy.concatenate(
            [pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
             pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
        org_h, org_w = org_img_shape
        resize_ratio = min(input_size / org_w, input_size / org_h)
        dw = (input_size - resize_ratio * org_w) / 2
        dh = (input_size - resize_ratio * org_h) / 2
        pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
        pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio
        pred_coor = numpy.concatenate(
            [numpy.maximum(pred_coor[:, :2], [0, 0]),
             numpy.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])],
            axis=-1)
        invalid_mask = numpy.logical_or((pred_coor[:, 0] > pred_coor[:, 2]),
                                        (pred_coor[:, 1] > pred_coor[:, 3]))
        pred_coor[invalid_mask] = 0
        bboxes_scale = numpy.sqrt(numpy.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
        scale_mask = numpy.logical_and((valid_scale[0] < bboxes_scale),
                                       (bboxes_scale < valid_scale[1]))
        classes = numpy.argmax(pred_prob, axis=-1)
        scores = pred_conf * pred_prob[numpy.arange(len(pred_coor)), classes]
        score_mask = scores > score_threshold
        mask = numpy.logical_and(scale_mask, score_mask)
        coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]
        return numpy.concatenate([coors, scores[:, numpy.newaxis], classes[:, numpy.newaxis]], axis=-1)

    @staticmethod
    def bboxes_iou(boxes1, boxes2):
        """获取真实框和预测框的交并比
        """
        boxes1 = numpy.array(boxes1)
        boxes2 = numpy.array(boxes2)
        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        left_up = numpy.maximum(boxes1[..., :2], boxes2[..., :2])  # 选出最大值
        right_down = numpy.minimum(boxes1[..., 2:], boxes2[..., 2:])
        inter_section = numpy.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        ious = numpy.maximum(1.0 * inter_area / union_area, numpy.finfo(numpy.float32).eps)
        return ious

    def nms(self, bboxes, iou_threshold, sigma=0.3, method='nms'):
        """非极大值抑制：将刚刚提取出来的信息进行筛选，返回最好的预测值
        """
        classes_in_img = list(set(bboxes[:, 5]))
        best_bboxes = []
        for cls in classes_in_img:
            cls_mask = (bboxes[:, 5] == cls)
            cls_bboxes = bboxes[cls_mask]
            while len(cls_bboxes) > 0:
                max_ind = numpy.argmax(cls_bboxes[:, 4])
                best_bbox = cls_bboxes[max_ind]
                best_bboxes.append(best_bbox)
                cls_bboxes = numpy.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
                iou = self.bboxes_iou(best_bbox[numpy.newaxis, :4], cls_bboxes[:, :4])
                weight = numpy.ones((len(iou),), dtype=numpy.float32)
                assert method in ['nms', 'soft-nms']
                if method == 'nms':
                    iou_mask = iou > iou_threshold
                    weight[iou_mask] = 0.0
                if method == 'soft-nms':
                    weight = numpy.exp(-(1.0 * iou ** 2 / sigma))
                cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
                score_mask = cls_bboxes[:, 4] > 0.
                cls_bboxes = cls_bboxes[score_mask]
        return best_bboxes

    def get_if_watermark_frame(self):
        """ 水印检测
        :return:
        """
        image_data = self.image_preporcess(numpy.copy(self.image_data), [self.image_size, self.image_size])
        image_data_list = image_data[numpy.newaxis, :].tolist()
        headers = {"Content-type": "application/json"}
        try_times = 0
        while try_times < 3:
            try:
                response = requests.post(self.__watermark_frame_check_server_url, headers=headers,
                                         data=json.dumps({"signature_name": "predict",
                                                          "instances": image_data_list})).json()
                output = numpy.array(response['predictions'])
                output = numpy.reshape(output, (-1, 11))  # 6类+1可能性+4个坐标
                original_image_size = self.image_data.shape[:2]
                bboxes = self.postprocess_boxes(output, original_image_size, self.image_size, 0.7)
                bboxes = self.nms(bboxes, 0.45, method='nms')
                for i, bbox in enumerate(bboxes):
                    class_ind = int(bbox[5])
                    self.result_list.append(self.watermark_classes[class_ind])
                return self.result_list
            except Exception as err:
                print(err)
                try_times += 1
                continue
        if try_times >= 3:
            return -1


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
        clrs = self.image_data.getcolors(maxcolors=180 * 255 * 255)  # maxcolors修改为HSV组合的最大数
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


class BlackWhiteImage(object):
    """黑白屏类
    """

    def __init__(self, img: numpy.ndarray = None):
        """
        :param img: 输入图像
        """
        self.image_data = Image.fromarray(img)
        self.image_data = self.image_data.convert('HSV')
        self.width, self.height = self.image_data.size

    def get_black_white_frame(self):
        clrs = self.image_data.getcolors(maxcolors=180 * 255 * 255)  # maxcolors修改为HSV组合的最大数
        blacks = 0
        whites = 0
        if clrs:
            for clr in clrs:
                if 0 <= clr[1][2] <= 46:
                    blacks += clr[0]
                if 0 <= clr[1][1] <= 30 and 221 <= clr[1][2] <= 255:
                    whites += clr[0]
            res = {"black_ratio": blacks / (self.width * self.height),
                   "white_ratio": whites / (self.width * self.height)}
            return res
        else:
            return None


class ImageColorLayer(object):
    def __init__(self, img: numpy.ndarray = None):
        self.img = img
        self.m, self.n, self.c = self.img.shape
        self.hsv = cv2.cvtColor(self.img, cv2.COLOR_RGB2HSV)
        self.red_lower1 = numpy.array([0, 40, 43])
        self.red_upper1 = numpy.array([10, 255, 255])
        self.red_lower2 = numpy.array([153, 40, 43])
        self.red_upper2 = numpy.array([180, 255, 255])
        self.green_lower = numpy.array([35, 40, 43])
        self.green_upper = numpy.array([99, 255, 255])
        self.blue_lower = numpy.array([100, 40, 43])
        self.blue_upper = numpy.array([130, 255, 255])
        self.color_list = [0, 1, 2]
        self.color_map = numpy.zeros((self.m, self.n), numpy.int8)
        self.color_dict = {'blue': '', 'green': '', 'red': '', 'exist_deep_red': False}

    def get_colorlayer_info(self):
        self.__get_red_ratio()
        self.__get_colors_ratio()
        return self.color_dict

    def __get_red_ratio(self):
        """计算具有固定数值的图层颜色所占比例，针对深红色用于判断有无
        """
        xy = numpy.where(
            (self.img[:, :, 0] >= 220) & (self.img[:, :, 0] <= 255) &
            (self.img[:, :, 1] >= 108) & (self.img[:, :, 1] <= 148) &
            (self.img[:, :, 2] >= 108) & (self.img[:, :, 2] <= 148),
            1, -1)
        deep_red_area = numpy.sum(xy == 1)
        deep_red_ratio = deep_red_area / (self.m * self.n)
        deep_red_flag = False
        if deep_red_ratio < 0.003:  # 深红占比少于0.3%认为不存在
            pass
        else:
            deep_red_flag = True
        self.color_dict['exist_deep_red'] = deep_red_flag

    def __get_colors_ratio(self):
        """计算红绿蓝图层的比例
        """
        for color_index in self.color_list:
            if color_index == 0:  # 浅红色
                mask1 = cv2.inRange(self.hsv, self.red_lower1, self.red_upper1)
                mask2 = cv2.inRange(self.hsv, self.red_lower2, self.red_upper2)
                mask = cv2.bitwise_or(mask1, mask2)
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    cv2.fillPoly(mask, [cnt], (255, 255, 255))
                self.color_map = numpy.where(mask != 0, 1, 0)
                red_area = numpy.sum(self.color_map == 1)
                red_ratio = '{:.2%}'.format(red_area / (self.m * self.n))
                self.color_dict['red'] = red_ratio
            elif color_index == 1:  # 绿色
                mask = cv2.inRange(self.hsv, self.green_lower, self.green_upper)
                cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                self.color_map = numpy.where((mask != 0) & (self.color_map == 0), 2, 0)
                green_area = numpy.sum(self.color_map == 2)
                red_ratio = '{:.2%}'.format(green_area / (self.m * self.n))
                self.color_dict['green'] = red_ratio
            elif color_index == 2:  # 蓝色
                mask = cv2.inRange(self.hsv, self.blue_lower, self.blue_upper)
                cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                self.color_map = numpy.where((mask != 0) & (self.color_map == 0), 3, 0)
                blue_area = numpy.sum(self.color_map == 3)
                red_ratio = '{:.2%}'.format(blue_area / (self.m * self.n))
                self.color_dict['blue'] = red_ratio


class ImageMatcher(object):
    """ 通用图像匹配类， 基于sift/surf算法
    """

    def __init__(self, template_image, target_image):
        """
        param: template_image: 模版图像，即子图
        param: target_image: 匹配图像，即被匹配图像
        """
        # 颜色空间转化
        self.__template_image = cv2.cvtColor(template_image, cv2.COLOR_RGB2BGR)
        self.__target_image = cv2.cvtColor(target_image, cv2.COLOR_RGB2BGR)
        # const
        self.__MIN_MATCH_COUNT = 10
        self.__FLANN_INDEX_KDTREE = 1

    def __calc_central_coordinates(self, coordinate_list: list):
        """
        """
        pass

    def get_template_image_coordinates(self):
        """
        """
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(self.__template_image, None)
        kp2, des2 = sift.detectAndCompute(self.__target_image, None)
        index_params = dict(algorithm=self.__FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
        if len(good) > self.__MIN_MATCH_COUNT:
            src_pts = numpy.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = numpy.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            h, w, _ = self.__template_image.shape
            pts = numpy.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            central_coordinates = numpy.int32(dst.mean(0)).tolist()
            if len(central_coordinates) >= 1:
                central_coordinates = central_coordinates[0]
            else:
                central_coordinates = []
        else:
            central_coordinates = []

        # 验证测试代码
        # cv2.circle(self.__target_image, tuple(central_coordinates), 10, (0, 0, 255), 4)
        # cv2.imwrite('test.png', self.__target_image)

        return {"match_coordinates": central_coordinates}


class ORBSimilarity(object):
    """
    利用ORB算法计算两张图像的相似性
    """
    def __init__(self, src_image, target_image):
        self.__src_image = cv2.cvtColor(src_image, cv2.COLOR_RGB2GRAY)
        self.__target_image = cv2.cvtColor(target_image, cv2.COLOR_RGB2GRAY)

    def get_similarity(self):
        finder = cv2.ORB_create()
        kp1, des1 = finder.detectAndCompute(self.__src_image, None)
        kp2, des2 = finder.detectAndCompute(self.__target_image, None)
        bf = cv2.BFMatcher(normType=cv2.NORM_HAMMING)
        matches = bf.knnMatch(des1, trainDescriptors=des2, k=2)

        goodMatch = []
        for m, n in matches:
            # goodMatch是经过筛选的优质配对，如果2个配对中第一匹配的距离小于第二匹配的距离的7/10，
            # 基本可以说明这个第一配对是两幅图像中独特的，不重复的特征点,可以保留。
            if m.distance < 0.7 * n.distance:
                goodMatch.append(m)
        l1 = len(goodMatch)
        l2 = len(matches)
        if l2 == 0:
            return -1
        else:
            similary = l1 / l2
            return similary


class ImageQualityIndexGenerator(object):
    """ 图像质量指标库
    """

    def __init__(self, image_file, target_image_file=None):
        """
        param: image_file: 必要输入的图像
        param: target_image_file: 多图像可选项
        """
        self.image_data = self.__bytesIO2img(image_file)
        if target_image_file:
            self.target_image_file = self.__bytesIO2img(target_image_file)
        else:
            self.target_image_file = None

    def __bytesIO2img(self, image_file):
        """
        :return:
        """
        img = numpy.frombuffer(image_file, dtype=numpy.uint8)
        img = cv2.imdecode(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

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
        __img_std = CnStd()
        __img_ocr = CnOcr(name=str(random.random()))
        self.image_data = cv2.cvtColor(self.image_data, cv2.COLOR_BGR2RGB)
        box_info_list = __img_std.detect(self.image_data)
        ocr_result_list = []
        for box_info in box_info_list:
            cropped_img = box_info['cropped_img']
            ocr_res = __img_ocr.ocr_for_single_line(cropped_img)
            ocr_result_list.append({'text': ''.join(ocr_res), 'coordinate': numpy.mean(box_info['box'], axis=0).tolist()})
        del __img_std
        del __img_ocr
        return ocr_result_list

    def get_if_blurred_frame(self):
        """ 花屏检测
        :return:
        """
        blurred_detect_handler = BlurredFrameDetector(self.image_data)
        return blurred_detect_handler.get_if_blurred_frame()

    def get_image_ssim(self):
        """ 图像结构相似度相似度
        :return:
        """
        image_similarity_handler = ORBSimilarity(self.image_data,self.target_image_file)
        return image_similarity_handler.get_similarity()

    def get_image_clarity(self):
        image_clarity_handler = ImageClarity(self.image_data)
        return image_clarity_handler.get_frame_clarity_nrss()

    def get_green_image(self):
        green_image_handler = GreenImage(self.image_data)
        return green_image_handler.get_green_frame()

    def get_black_white_image(self):
        black_white_image_handler = BlackWhiteImage(self.image_data)
        return black_white_image_handler.get_black_white_frame()

    def get_image_colorlayer(self):
        image_colorlayer_handler = ImageColorLayer(self.image_data)
        return image_colorlayer_handler.get_colorlayer_info()

    def get_image_watermark(self):
        image_watermark_handler = WatermarkFrameDetector(self.image_data)
        return image_watermark_handler.get_if_watermark_frame()

    def get_image_match_result(self):
        image_matcher_handler = ImageMatcher(template_image=self.image_data, target_image=self.target_image_file)
        return image_matcher_handler.get_template_image_coordinates()


if __name__ == '__main__':
    pass
