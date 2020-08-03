"""
打点数据预测封装类
通过调用该类获取视频播放质量指标
"""
from app.factory import MyMongoClient
from app.factory import LogManager


class DotVideoIndex(object):
    """
    """

    def __init__(self, video_dict: dict = {}):
        """
        :param video_dict:
        """
        self.__logger = LogManager("server.log").logger
        self.__video_dict = video_dict
        player_event_dict = self.__get_event_info()
        self.event_dict = {
            "main.ijk.will_prepare.tracker":
                player_event_dict.get("main.ijk.will_prepare.tracker", None),
            "main.ijk.replace_item.tracker":
                player_event_dict.get("main.ijk.replace_item.tracker", None),
            "main.ijk.seek_first_video_render.tracker":
                player_event_dict.get("main.ijk.seek_first_video_render.tracke", None),
            "main.ijk.seek_first_audio_render.tracker":
                player_event_dict.get("main.ijk.seek_first_audio_render.tracker", None),
            "main.ijk.dash_did_switch_qn.tracker":
                player_event_dict.get("main.ijk.dash_did_switch_qn.tracker", None),  #
            "main.ijk.set_auto_switch.tracker":
                player_event_dict.get("main.ijk.set_auto_switch.tracker", None),  # dash自动切换
            "main.ijk.buffering_start.tracker":
                player_event_dict.get("main.ijk.buffering_start.tracker", None),  # buffering start
            "main.ijk.buffering_end.tracker":
                player_event_dict.get("main.ijk.buffering_end.tracker", None),  # buffering end
            "main.ijk.decode_switch.tracker":
                player_event_dict.get("main.ijk.decode_switch.tracker", None),  # 解码切换
            "main.ijk.http_build.tracker":
                player_event_dict.get("main.ijk.http_build.tracker", None),  # http 建联 + range
            "main.ijk.asset_item_start.tracker":
                player_event_dict.get("main.ijk.asset_item_start.tracker", None),  # item 开始工作
            "main.ijk.asset_item_stop.tracker":
                player_event_dict.get("main.ijk.asset_item_stop.tracker", None),  # 停止播放
            "main.ijk.first_video_render.tracker":
                player_event_dict.get("main.ijk.first_video_render.tracker", None)  # 视频渲染首帧
        }

    def __get_event_info(self):
        """根据buvid从数据库获取各个播放事件的最新数据
        :return:
        """
        buvid = self.__video_dict.get("buvid", None)
        if buvid:
            return
        else:
            return {}

    def get_video_info(self):
        """获取该视频的其他次重要信息，例如视频时长，音频时长
         先暂时用播放stop事件的打点数据来获取需要的信息
        :return:
        """
        asset_item_stop_info = self.event_dict.get("main.ijk.asset_item_stop.tracker")
        if asset_item_stop_info:
            video_duration = asset_item_stop_info.get("video_duration", "")
            audio_duration = asset_item_stop_info.get("audio_duration", "")
            video_bitrate = asset_item_stop_info.get("video_bitrate", "")
            audio_bitrate = asset_item_stop_info.get("audio_bitrate", "")
            return video_duration, audio_duration, video_bitrate, audio_bitrate
        else:
            return None, None, None, None

    def get_first_av_time(self):
        """ 获取音视频首帧耗时
        :return:
        """
        asset_item_stop_info = self.event_dict.get("main.ijk.asset_item_stop.tracker")
        if asset_item_stop_info:
            first_video_time = self.event_dict.get("first_video_time", "")
            first_audio_time = self.event_dict.get("first_audio_time", "")
            return first_video_time, first_audio_time
        else:
            return None, None

    def get_freeze_rate(self):
        """获取丢帧率
        :return:
        """
        asset_item_stop_info = self.event_dict.get("main.ijk.asset_item_stop.tracker")
        if asset_item_stop_info:
            freeze_rate = self.event_dict.get("vdrop_rate", "")
            return freeze_rate
        else:
            return None

    def get_audio_pts_diff(self):
        """获取音频播放偏差
        :return:
        """
        asset_item_stop_info = self.event_dict.get("main.ijk.asset_item_stop.tracker")
        if asset_item_stop_info:
            audio_pts_diff_time = self.event_dict.get("audio_pts_diff", "")
            return audio_pts_diff_time
        else:
            return None

    def get_asset_update_count(self):
        """获取该播放行为总资源刷新次数
        :return:
        """
        asset_item_stop_info = self.event_dict.get("main.ijk.asset_item_stop.tracker")
        if asset_item_stop_info:
            asset_update_count = self.event_dict.get("asset_update_count", "")
            return asset_update_count
        else:
            return None

    def get_ijk_cpu_mem_rate(self):
        """获取ijk进程cpu占用率, 内存占用率
        :return:
        """
        asset_item_stop_info = self.event_dict.get("main.ijk.asset_item_stop.tracker")
        if asset_item_stop_info:
            ijk_cpu_rate = self.event_dict.get("ijk_cpu_rate", "")
            ijk_mem = self.event_dict.get("ijk_mem", "")
            return ijk_cpu_rate, ijk_mem
        else:
            return None, None

    def get_black_screen_rate(self):
        """
        :return:
        """
        return

    def get_error_rate(self):
        """
        :return:
        """
        return

    def get_total_index(self):
        """所有指标获取
        :return:
        """
        video_duration, audio_duration, video_bitrate, audio_bitrate = self.get_video_info()
        first_video_time, first_audio_time = self.get_first_av_time()
        freeze_rate = self.get_freeze_rate()
        asset_update_count = self.get_asset_update_count()
        audio_pts_diff_time = self.get_audio_pts_diff()
        ijk_cpu_rate, ijk_mem = self.get_ijk_cpu_mem_rate()
        return {
            "video_duration": video_duration,
            "audio_duration": audio_duration,
            "video_bitrate": video_bitrate,
            "audio_bitrate": audio_bitrate,
            "first_video_time": first_video_time,
            "first_audio_time": first_video_time,
            "freeze_rate": freeze_rate,
            "asset_update_count": asset_update_count,
            "audio_pts_diff_time": audio_pts_diff_time,
            "ijk_cpu_rate": ijk_cpu_rate,
            "ijk_mem": ijk_mem
        }
