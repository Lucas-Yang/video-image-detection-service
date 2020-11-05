"""
打点数据预测封装类
通过调用该类获取视频播放质量指标
"""
import json
import requests
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
        """根据buvid从uat es接口获取各个播放事件的最新数据,
        :return:
        """
        es_url = "http://172.22.33.113:81/v7/billions-datacenter.buryingpoint.buryingpoint-@*/_search"
        payload = {
            "from": 0,
            "size": 1000,
            "query": {
                "bool": {
                    "filter": [
                        {
                            "match_phrase": {
                                "fields.env": {
                                    "query": "prod"
                                }
                            }
                        },
                        {
                            "query_string": {
                                "query": "\"logid:002879\""
                            }
                        },
                        {
                            "query_string": {
                                "query": self.__video_dict.get("device_id")
                            }
                        },
                        {
                            "query_string": {
                                "query": self.__video_dict.get("buvid")
                            }
                        },
                        {
                            "range": {
                                "@timestamp": {
                                    "gte": int(self.__video_dict.get("start_time")),
                                    "lte": int(self.__video_dict.get("end_time"))
                                }
                            }
                        }
                    ]
                }
            }
        }
        headers = {
            'Content-Type': 'application/json',
            'Appid': 'billions',
            'Appkey': 'proxy'
        }
        player_event_dict = {}
        retry_time = 0
        es_result_json = {}
        while retry_time < 3:
            try:
                r = requests.post(url=es_url, json=payload, headers=headers)
                r.raise_for_status()
                es_result_json = r.json()
                break
            except BaseException as error:
                self.__logger.error(error)
                retry_time += 1
        if retry_time >= 3:
            raise Exception("access ops-log error time > 3")
        else:
            try:
                for event_log in es_result_json.get("hits").get("hits"):
                    event_log = event_log.get("_source").get("log")
                    if 'kafka-consumer' not in event_log:
                        continue
                    else:
                        event_info_list = event_log.split("||||")
                        event_name = event_info_list[0].split("|")[-5]
                        event_info_json = json.loads(event_info_list[1])
                        if event_name in player_event_dict:
                            player_event_dict[event_name].append(event_info_json)
                        else:
                            player_event_dict[event_name] = [event_info_json]
            except BaseException as error:
                self.__logger.error(error)
                raise Exception('ops-log es port return is error')
        # print(json.dumps(player_event_dict))
        return player_event_dict

    def get_video_info(self):
        """获取该视频的其他次重要信息，例如视频时长，音频时长
         先暂时用播放stop事件的打点数据来获取需要的信息
        :return:
        """
        asset_item_stop_info = self.event_dict.get("main.ijk.asset_item_stop.tracker")
        if asset_item_stop_info:
            video_duration = asset_item_stop_info[0].get("video_duration", "")
            audio_duration = asset_item_stop_info[0].get("audio_duration", "")
            video_bitrate = asset_item_stop_info[0].get("video_bitrate", "")
            audio_bitrate = asset_item_stop_info[0].get("audio_bitrate", "")
            return video_duration, audio_duration, video_bitrate, audio_bitrate
        else:
            return None, None, None, None

    def get_first_av_time(self):
        """ 获取音视频首帧耗时
        :return:
        """
        asset_item_stop_info = self.event_dict.get("main.ijk.asset_item_stop.tracker")
        if asset_item_stop_info:
            first_video_time = asset_item_stop_info[0].get("first_video_time", "")
            first_audio_time = asset_item_stop_info[0].get("first_audio_time", "")
            return first_video_time, first_audio_time
        else:
            return None, None

    def get_frame_loss_rate(self):
        """获取丢帧率
        :return:
        """
        asset_item_stop_info = self.event_dict.get("main.ijk.asset_item_stop.tracker")
        if asset_item_stop_info:
            freeze_rate = asset_item_stop_info[0].get("vdrop_rate", "")
            return freeze_rate
        else:
            return None

    def get_freeze_times(self):
        """获取卡顿次数
        :return:
        """
        asset_item_stop_info = self.event_dict.get("main.ijk.asset_item_stop.tracker")
        if asset_item_stop_info:
            freeze_times = asset_item_stop_info[0].get("buffering_count", "")
            return freeze_times
        else:
            return None

    def get_freeze_rate(self):
        """获取卡顿率，buffering时间 / 总播放时间
        :return:
        """
        asset_item_stop_info = self.event_dict.get("main.ijk.asset_item_stop.tracker")
        if asset_item_stop_info:
            buffering_time = self.get_buffering_total_time()
            asset_item_time_of_session = float(asset_item_stop_info[0].get("asset_item_time_of_session", 0))
            if asset_item_time_of_session == 0.0:
                raise Exception("get freeze rate error, asset session time is 0")
            else:
                return buffering_time / asset_item_time_of_session if buffering_time else 0
        else:
            return None

    def get_audio_pts_diff(self):
        """获取音频播放偏差
        :return:
        """
        asset_item_stop_info = self.event_dict.get("main.ijk.asset_item_stop.tracker")
        if asset_item_stop_info:
            audio_pts_diff_time = asset_item_stop_info[0].get("audio_pts_diff", "")
            return audio_pts_diff_time
        else:
            return None

    def get_asset_update_count(self):
        """获取该播放行为总资源刷新次数
        :return:
        """
        asset_item_stop_info = self.event_dict.get("main.ijk.asset_item_stop.tracker")
        if asset_item_stop_info:
            asset_update_count = asset_item_stop_info[0].get("asset_update_count", "")
            return asset_update_count
        else:
            return None

    def get_buffering_total_time(self):
        """获取缓冲耗时
        :return:
        """
        asset_item_buffering_end_info_list = self.event_dict.get("main.ijk.buffering_end.tracker")
        if asset_item_buffering_end_info_list:
            total_time = 0
            for event_info in asset_item_buffering_end_info_list:
                total_time += float(event_info.get("time_of_event"))
            return total_time
        else:
            return None

    def get_video_read_bytes(self):
        """获取音视频读取字节数
        :return:
        """
        asset_item_stop_info = self.event_dict.get("main.ijk.asset_item_stop.tracker")
        if asset_item_stop_info:
            video_read_bytes = asset_item_stop_info[0].get("video_read_bytes", "")
            audio_read_bytes = asset_item_stop_info[0].get("audio_read_bytes", "")
            return video_read_bytes, audio_read_bytes
        else:
            return None, None

    def get_ijk_cpu_mem_rate(self):
        """获取ijk进程cpu占用率, 内存占用率
        :return:
        """
        asset_item_stop_info = self.event_dict.get("main.ijk.asset_item_stop.tracker")
        if asset_item_stop_info:
            ijk_cpu_rate = asset_item_stop_info[0].get("ijk_cpu_rate", "")
            ijk_mem = asset_item_stop_info[0].get("ijk_mem", "")
            return ijk_cpu_rate, ijk_mem
        else:
            return None, None

    def get_black_screen_rate(self):
        """
        :return:
        """
        return

    def get_error(self):
        """
        :return:
        """
        asset_item_stop_info = self.event_dict.get("main.ijk.asset_item_stop.tracker")
        if asset_item_stop_info:
            last_video_net_error = asset_item_stop_info[0].get("last_video_net_error", "")
            last_audio_net_error = asset_item_stop_info[0].get("last_audio_net_error", "")
            exit_player_status = asset_item_stop_info[0].get("player_status", "")
            return last_audio_net_error, last_video_net_error, exit_player_status
        else:
            return None, None, None

    def get_total_index(self):
        """所有指标获取
        :return:
        """
        video_duration, audio_duration, video_bitrate, audio_bitrate = self.get_video_info()
        first_video_time, first_audio_time = self.get_first_av_time()
        frame_loss_rate = self.get_frame_loss_rate()
        freeze_times = self.get_freeze_times()
        freeze_rate = self.get_freeze_rate()
        video_read_bytes, audio_read_bytes = self.get_video_read_bytes()
        asset_update_count = self.get_asset_update_count()
        audio_pts_diff_time = self.get_audio_pts_diff()
        ijk_cpu_rate, ijk_mem = self.get_ijk_cpu_mem_rate()
        buffering_total_time = self.get_buffering_total_time()
        audio_error_code, video_error_code, exit_player_status = self.get_error()

        return {
            "video_base_info": {"video_duration": video_duration,
                                "audio_duration": audio_duration,
                                "video_bitrate": video_bitrate,
                                "audio_bitrate": audio_bitrate,
                                "video_read_bytes": video_read_bytes,
                                "audio_read_bytes": audio_read_bytes
                                },
            "exit_error_info": {
                "last_audio_net_error_code": audio_error_code,
                "last_video_net_error_code": video_error_code,
                "exit_player_status": exit_player_status
            },
            "first_video_time": first_video_time,
            "first_audio_time": first_video_time,
            "frame_loss_rate": frame_loss_rate,
            "freeze_times": freeze_times,
            "freeze_rate": freeze_rate,
            "buffering_total_time": buffering_total_time,
            "asset_update_count": asset_update_count,
            "audio_pts_diff_time": audio_pts_diff_time,
            "ijk_cpu_rate": ijk_cpu_rate,
            "ijk_mem": ijk_mem
        }


if __name__ == '__main__':
    video_info = {"device_id": "awhuXmdXNVFlBGVTL1Mv",
                  "buvid": "XYC4F53C2D90023964D4CDF500BFA73C5BC19",
                  "start_time": "1597656191475",
                  "end_time": "1597656852622"
                  }
    dot_handler = DotVideoIndex(video_dict=video_info)
    print(json.dumps(dot_handler.get_total_index()))
