"""
全参考视频质量评估，包括SSIM, PSNR, VMAF
"""
import os
import re
import subprocess
import sys

import cv2
import ffmpeg
from skimage.metrics import structural_similarity as compare_ssim


class VideoSSIM(object):
    """结构相似性评估类
    """

    def __init__(self, src_video=None, target_video=None):
        self.src_video = src_video
        self.target_video = target_video

    @staticmethod
    def __get_video_info(video_path):
        """
        opencv 获取视频基本信息
        :param video_path:
        :return:
        :return:
        """
        cap = cv2.VideoCapture(video_path)
        total_frame = cap.get(7)  # 帧数
        fps = cap.get(5)  # 帧率
        per_frame_time = 1 / fps
        return total_frame, fps, per_frame_time

    def __cut_frame(self, video_path):
        """
        :return:
        """
        cap = cv2.VideoCapture(video_path)
        # total_frame, fps, per_frame_time = self.__get_video_info(video_path)
        count = 0
        success = True
        while success:
            count += 1
            success, frame = cap.read()
            if success:
                yield frame
            else:
                continue
        cap.release()

    @staticmethod
    def frame_ssim(frame_src, frame_target):
        """ 获取单帧ssim值
        :param frame_target:
        :param frame_src:
        :return:
        """
        gray_src = cv2.cvtColor(frame_src, cv2.COLOR_BGR2GRAY)
        gray_target = cv2.cvtColor(frame_target, cv2.COLOR_BGR2GRAY)
        (score, diff) = compare_ssim(gray_src, gray_target, full=True)
        return score

    def get_video_ssim_index(self):
        """对外接口，获取两个视频的结构相似度指标，并获取相似性小于95%的帧
        :return:
        """
        total_ssim_score = 0
        frame_counts = 0
        for frame_src, frame_target in zip(self.__cut_frame(self.src_video), self.__cut_frame(self.target_video)):
            ssim_score = self.frame_ssim(frame_src, frame_target)
            total_ssim_score += ssim_score
            frame_counts += 1
            print("SSIM: {}".format(ssim_score))
            del ssim_score
        if frame_counts != 0:
            video_ssim_score = total_ssim_score / frame_counts
        else:
            video_ssim_score = 0
        return video_ssim_score

    def get_video_ffmpeg_ssim_index(self):
        """ 根据ffmpeg计算ssim
        :return:
        """
        stream1 = ffmpeg.input(self.src_video)
        stream2 = ffmpeg.input(self.target_video)
        stream = ffmpeg.filter_([stream1, stream2], 'ssim')
        stream = ffmpeg.output(stream, 'pipe:', format='null')
        out, line_info = stream.run(quiet=True, capture_stdout=True)
        match_objects_start = re.match(".* All:(.*) .*", bytes.decode(line_info).strip().split('\n')[-1], re.M | re.I)
        if match_objects_start:
            video_ssim_score = float(match_objects_start.group(1))
        else:
            video_ssim_score = 0
        return video_ssim_score

    def test(self):
        """
        :return:
        """
        for frame in self.__cut_frame('/Users/luoyadong/Desktop/studio_video_1605840496434.mp4'):
            cv2.imshow("test", frame)
            cv2.waitKey(1000)


class VideoVMAF(object):
    """主观质量评估类
    """
    current_dirname = os.path.dirname(os.path.realpath(__file__))

    def __init__(self, input_video=None, refer_video=None,
                 model_path=os.path.join(current_dirname, "vmaf_v0.6.1.json")):
        self.input_video = input_video
        self.refer_video = refer_video
        self.model_path = model_path

    def get_video_vmaf_score(self) -> str:
        """ 根据ffmpeg计算vmaf
        :return:
        """
        stream1 = ffmpeg.input(self.input_video)
        stream2 = ffmpeg.input(self.refer_video)
        stream = ffmpeg.filter_([stream1, stream2], 'libvmaf', model_path=self.model_path)
        p = subprocess.Popen(stream.output('-', format='null').compile() + ['-nostats'],
                             stderr=subprocess.PIPE)
        output = p.communicate()[1].decode('utf-8')
        if p.returncode != 0:
            sys.stderr.write(output)
            return None
        lines = output.splitlines()
        vmaf_score = lines[-1].split("score:")[-1].strip()  # vmaf的分数在命令行输出的最后一行
        return vmaf_score

class VideoPSNR(object):

    def __init__(self, input_video=None, refer_video=None):
        self.input_video = input_video
        self.refer_video = refer_video

    def get_video_psnr_score(self) -> str:
        """ 根据ffmpeg计算psnr
        :return: 视频的psnr平均分
        """
        stream1 = ffmpeg.input(self.input_video)
        stream2 = ffmpeg.input(self.refer_video)
        stream = ffmpeg.filter_([stream1, stream2], 'psnr')
        p = subprocess.Popen(stream.output('-', format='null').compile() + ['-nostats'],
                             stderr=subprocess.PIPE)
        output = p.communicate()[1].decode('utf-8')
        if p.returncode != 0:
            sys.stderr.write(output)
            return None
        lines = output.splitlines()
        psnr_score = lines[-1].split(' ')[7].split(':')[1]
        return psnr_score

if __name__ == "__main__":
    ssim_handler = VideoSSIM("/Users/luoyadong/Desktop/studio_video_1605840496434.mp4",
                             "/Users/luoyadong/Desktop/studio_video_16058404964343.mp4")
    print(ssim_handler.get_video_ssim_index())