import time

import numpy as np
import scipy.io
from os.path import dirname
from os.path import join
from skvideo.utils import *
from skvideo import io
from PIL import Image

import scipy.special
import math
import cv2
from joblib import load

class VideoNIQE(object):
    def __init__(self, input_video=None):
        self.input_video = input_video

    def get_video_niqe_score(self) -> str:
        """ 改写skvideo中的方法计算niqe分数
        :return: 视频的niqe平均分
        """
        try:
            input_frames = io.vread(self.input_video, outputdict={"-pix_fmt": "gray"})[:, :, :, 0]
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
        """Computes Naturalness Image Quality Evaluator. [#f1]_

        Input a video of any quality and get back its distance frame-by-frame
        from naturalness.

        Parameters
        ----------
        inputVideoData : ndarray
            Input video, ndarray of dimension (T, M, N, C), (T, M, N), (M, N, C), or (M, N),
            where T is the number of frames, M is the height, N is width,
            and C is number of channels. Here C is only allowed to be 1.

        Returns
        -------
        niqe_array : ndarray
            The niqe results, ndarray of dimension (T,), where T
            is the number of frames

        References
        ----------

        .. [#f1] Mittal, Anish, Rajiv Soundararajan, and Alan C. Bovik. "Making a 'completely blind' image quality analyzer." IEEE Signal Processing Letters 20.3 (2013): 209-212.

        """
        # cache
        patch_size = 96
        module_path = dirname(__file__)

        params = scipy.io.loadmat(join(module_path, 'niqe_image_params.mat'))
        pop_mu = np.ravel(params["pop_mu"])
        pop_cov = params["pop_cov"]

        # load the training data
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

    @staticmethod
    def _niqe_extract_subband_feats(mscncoefs):
        # alpha_m,  = extract_ggd_features(mscncoefs)
        alpha_m, N, bl, br, lsq, rsq = aggd_features(mscncoefs.copy())
        pps1, pps2, pps3, pps4 = paired_product(mscncoefs)
        alpha1, N1, bl1, br1, lsq1, rsq1 = aggd_features(pps1)
        alpha2, N2, bl2, br2, lsq2, rsq2 = aggd_features(pps2)
        alpha3, N3, bl3, br3, lsq3, rsq3 = aggd_features(pps3)
        alpha4, N4, bl4, br4, lsq4, rsq4 = aggd_features(pps4)
        return np.array([alpha_m, (bl + br) / 2.0,
                         alpha1, N1, bl1, br1,  # (V)
                         alpha2, N2, bl2, br2,  # (H)
                         alpha3, N3, bl3, bl3,  # (D1)
                         alpha4, N4, bl4, bl4,  # (D2)
                         ])

    def get_patches_test_features(self, img, patch_size, stride=8):
        return self._get_patches_generic(img, patch_size, 0, stride)

    def extract_on_patches(self, img, patch_size):
        h, w = img.shape
        patch_size = np.int(patch_size)
        patches = []
        for j in range(0, h - patch_size + 1, patch_size):
            for i in range(0, w - patch_size + 1, patch_size):
                patch = img[j:j + patch_size, i:i + patch_size]
                patches.append(patch)

        patches = np.array(patches)

        patch_features = []
        for p in patches:
            patch_features.append(self._niqe_extract_subband_feats(p))
        patch_features = np.array(patch_features)

        return patch_features

    def _get_patches_generic(self, img, patch_size, is_train, stride):
        h, w = np.shape(img)
        if h < patch_size or w < patch_size:
            print("输入图像尺寸太小了！")
            exit(-1)

        # ensure that the patch divides evenly into img
        hoffset = (h % patch_size)
        woffset = (w % patch_size)

        if hoffset > 0:
            img = img[:-hoffset, :]
        if woffset > 0:
            img = img[:, :-woffset]

        img = img.astype(np.float32)
        # scipy1.3.0版本之后弃用了misc.imresize，如果沿用会报错，现改用PIL里的方法代替
        # img2 = scipy.misc.imresize(img, 0.5, interp='bicubic', mode='F')
        img2 = np.array(Image.fromarray(img).resize(
            (int(0.5 * img.shape[0]), int(img.shape[1] * 0.5)),
            resample=Image.BICUBIC)
        )

        mscn1, var, mu = compute_image_mscn_transform(img)
        mscn1 = mscn1.astype(np.float32)

        mscn2, _, _ = compute_image_mscn_transform(img2)
        mscn2 = mscn2.astype(np.float32)

        feats_lvl1 = self.extract_on_patches(mscn1, patch_size)
        feats_lvl2 = self.extract_on_patches(mscn2, patch_size / 2)

        feats = np.hstack((feats_lvl1, feats_lvl2))  # feats_lvl3))

        return feats


class VideoBRISQUE(object):
    def __init__(self,input_video):
        self.input_video = input_video
        gamma_range = np.arange(0.2, 10, 0.001)
        self.gamma_range = gamma_range
        a = scipy.special.gamma(2.0 / gamma_range)
        a *= a
        b = scipy.special.gamma(1.0 / gamma_range)
        c = scipy.special.gamma(3.0 / gamma_range)
        self.prec_gammas = a / (b * c)
        module_path = dirname(__file__)
        self.model_path = join(module_path, 'svr_brisque.joblib')

    def get_brisque_score(self):
        try:
            input_frames = io.vread(self.input_video, outputdict={"-pix_fmt": "gray"})[:, :, :, 0]
        except:
            print('读取视频出错！')
            return None
        else:
            frames_scores = self.brisque(input_frames)
            avg_scores = np.mean(frames_scores)
            #print(avg_scores)
            if avg_scores == -1:
                return None
            return str(avg_scores)

    def brisque(self, inputVideoData):
        inputVideoData = vshape(inputVideoData)
        T, M, N, C = inputVideoData.shape
        feats = np.zeros((T, 36))
        #print('T:',T)
        for t in range(T):
            mscncoefs = self.calculate_mscn(inputVideoData[t, :, :, 0])
            features1 = self.extract_brisque_feats(mscncoefs)
            lowResolution = cv2.resize(inputVideoData[t, :, :, 0], (0, 0), fx=0.5, fy=0.5)
            features2 = self.extract_brisque_feats(lowResolution)
            feature = np.array(features1 + features2)
            feats[t] = feature

        clf = load(self.model_path)
        scores = clf.predict(feats)
        return scores

    def aggd_features(self, imdata):
        # flatten imdata
        imdata.shape = (len(imdata.flat),)
        imdata2 = imdata * imdata
        left_data = imdata2[imdata < 0]
        right_data = imdata2[imdata >= 0]
        left_mean_sqrt = 0
        right_mean_sqrt = 0
        if len(left_data) > 0:
            left_mean_sqrt = np.sqrt(np.average(left_data))
        if len(right_data) > 0:
            right_mean_sqrt = np.sqrt(np.average(right_data))

        if right_mean_sqrt != 0:
            gamma_hat = left_mean_sqrt / right_mean_sqrt
        else:
            gamma_hat = np.inf
        # solve r-hat norm

        imdata2_mean = np.mean(imdata2)
        if imdata2_mean != 0:
            r_hat = (np.average(np.abs(imdata)) ** 2) / (np.average(imdata2))
        else:
            r_hat = np.inf
        rhat_norm = r_hat * (((math.pow(gamma_hat, 3) + 1) *
                              (gamma_hat + 1)) / math.pow(math.pow(gamma_hat, 2) + 1, 2))

        # solve alpha by guessing values that minimize ro
        pos = np.argmin((self.prec_gammas - rhat_norm) ** 2)
        alpha = self.gamma_range[pos]

        gam1 = scipy.special.gamma(1.0 / alpha)
        gam2 = scipy.special.gamma(2.0 / alpha)
        gam3 = scipy.special.gamma(3.0 / alpha)

        aggdratio = np.sqrt(gam1) / np.sqrt(gam3)
        bl = aggdratio * left_mean_sqrt
        br = aggdratio * right_mean_sqrt

        # mean parameter
        N = (br - bl) * (gam2 / gam1)  # *aggdratio
        return (alpha, N, bl, br, left_mean_sqrt, right_mean_sqrt)

    @staticmethod
    def paired_product(new_im):
        shift1 = np.roll(new_im.copy(), 1, axis=1)
        shift2 = np.roll(new_im.copy(), 1, axis=0)
        shift3 = np.roll(np.roll(new_im.copy(), 1, axis=0), 1, axis=1)
        shift4 = np.roll(np.roll(new_im.copy(), 1, axis=0), -1, axis=1)

        H_img = shift1 * new_im
        V_img = shift2 * new_im
        D1_img = shift3 * new_im
        D2_img = shift4 * new_im

        return (H_img, V_img, D1_img, D2_img)

    @staticmethod
    def calculate_mscn(dis_image):
        dis_image = dis_image.astype(np.float32)  # 类型转换十分重要
        ux = cv2.GaussianBlur(dis_image, (7, 7), 7 / 6)
        ux_sq = ux * ux
        sigma = np.sqrt(np.abs(cv2.GaussianBlur(dis_image ** 2, (7, 7), 7 / 6) - ux_sq))
        mscn = (dis_image - ux) / (1 + sigma)

        return mscn

    def ggd_features(self, imdata):
        nr_gam = 1 / self.prec_gammas
        sigma_sq = np.var(imdata)
        E = np.mean(np.abs(imdata))
        rho = sigma_sq / E ** 2
        pos = np.argmin(np.abs(nr_gam - rho))
        return self.gamma_range[pos], sigma_sq

    @staticmethod
    def extract_brisque_feats(mscncoefs):
        alpha_m, sigma_sq = ggd_features(mscncoefs.copy())
        pps1, pps2, pps3, pps4 = paired_product(mscncoefs)
        alpha1, N1, bl1, br1, lsq1, rsq1 = aggd_features(pps1)
        alpha2, N2, bl2, br2, lsq2, rsq2 = aggd_features(pps2)
        alpha3, N3, bl3, br3, lsq3, rsq3 = aggd_features(pps3)
        alpha4, N4, bl4, br4, lsq4, rsq4 = aggd_features(pps4)
        # print(alpha_m, alpha1)
        return [
            alpha_m, sigma_sq,
            alpha1, N1, lsq1 ** 2, rsq1 ** 2,  # (V)
            alpha2, N2, lsq2 ** 2, rsq2 ** 2,  # (H)
            alpha3, N3, lsq3 ** 2, rsq3 ** 2,  # (D1)
            alpha4, N4, lsq4 ** 2, rsq4 ** 2,  # (D2)
        ]

if __name__ == '__main__':
    # t1 = time.time()
    # v = VideoNIQE('../tests/video_data/vmaf_refer_video.mp4')
    # t2 = time.time()
    # print(v.get_video_niqe_score())
    # print(t2 - t1)

    b = VideoBRISQUE('../../tests/video_data/vmaf_input_video.mp4')
    t1 = time.time()
    b.get_brisque_score()
    t2 = time.time()
    print(t2-t1)
