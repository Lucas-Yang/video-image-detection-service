import time

import numpy as np
import scipy.io
from os.path import dirname
from os.path import join
from skvideo.utils import *
from skvideo import io

import scipy.special
import cv2
from joblib import load
from app.third_lib.no_reference_image_quality import ImageBRISQUE, ImageNIQE

class VideoNIQE(ImageNIQE):
    """
        继承了图像的ImageNIQE类作为父类以预测视频的NIQE分数
    """
    def get_video_niqe_score(self, input_video):
        try:
            input_frames = io.vread(input_video, outputdict={"-pix_fmt": "gray"})[:, :, :, 0]
        except:
            print('读取视频出错！')
            return None
        else:
            frames_scores = self.niqe(input_frames)
            avg_scores = np.mean(frames_scores)
            if avg_scores == -1:
                return None
            return str(avg_scores)

    def niqe(self, inputVideoData):
        patch_size = 96
        params = scipy.io.loadmat(self.model_path)
        pop_mu = np.ravel(params["pop_mu"])
        pop_cov = params["pop_cov"]
        inputVideoData = vshape(inputVideoData)

        T, M, N, C = inputVideoData.shape

        assert C == 1, "niqe called with videos containing %d channels. Please supply only the luminance channel" % (
        C,)
        assert M > (
                patch_size * 2 + 1), "niqe called with small frame size, requires > 192x192 resolution video using current training parameters"
        assert N > (
                patch_size * 2 + 1), "niqe called with small frame size, requires > 192x192 resolution video using current training parameters"

        niqe_scores = np.zeros(T, dtype=np.float32)

        for t in range(T):
            try:
                feats = self.get_patches_test_features(inputVideoData[t, :, :, 0], patch_size)
            except Exception as err:
                return -1
            sample_mu = np.mean(feats, axis=0)
            sample_cov = np.cov(feats.T)
            X = sample_mu - pop_mu
            covmat = ((pop_cov + sample_cov) / 2.0)
            pinvmat = scipy.linalg.pinv(covmat)
            niqe_scores[t] = np.sqrt(np.dot(np.dot(X, pinvmat), X))
        return niqe_scores


class VideoBRISQUE(ImageBRISQUE):
    """
    继承了图像的ImageBRISQUE类作为父类以预测视频的BRISQUE分数
    """
    def get_video_brisque_score(self, input_video):
        try:
            # 用skvideo读取视频进行切帧并转为灰度图像
            input_frames = io.vread(input_video, outputdict={"-pix_fmt": "gray"})[:, :, :, 0]
        except:
            print('读取视频出错！')
            return None
        else:
            # 将每一帧图像传入计算每一帧的BRISQUE分数
            frames_scores = self.brisque(input_frames)
            # 取所有帧的平均分作为视频的最终分数
            avg_scores = np.mean(frames_scores)
            if avg_scores == -1:
                return None
            return str(avg_scores)

    def brisque(self, inputVideoData):
        inputVideoData = vshape(inputVideoData)
        T, M, N, C = inputVideoData.shape
        feats = np.zeros((T, 36))
        for t in range(T):
            # 计算得到图像的mscn系数
            mscncoefs = self.calculate_mscn(inputVideoData[t, :, :, 0])
            # 利用mscn计算18维的特征向量
            features1 = self.extract_brisque_feats(mscncoefs)
            # 将图像缩小到原始大小的一半，并重复相同的过程以获得一组新的特征
            halfimg = cv2.resize(inputVideoData[t, :, :, 0], (0, 0), fx=0.5, fy=0.5)
            lowResolution = self.calculate_mscn(halfimg)
            features2 = self.extract_brisque_feats(lowResolution)
            feature = np.array(features1 + features2)
            feats[t] = feature
        # 加载训练好的模型对视频的每一帧进行预测打分
        clf = load(self.model_path)
        scores = clf.predict(feats)
        return scores


if __name__ == '__main__':
    #b = VideoBRISQUE()
    n = VideoNIQE()
    t1 = time.time()
    #print(b.get_video_brisque_score('../../tests/video_data/vmaf_input_video.mp4'))
    print(n.get_video_niqe_score('../../tests/video_data/vmaf_input_video.mp4'))
    t2 = time.time()
    print(t2-t1)

