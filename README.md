# 播放器播放指标服务

## 一 概述
本服务用于获取播放视频的质量指标。
后端用两种方式来计算播放指标：端上播放器打点数据获取，cv方式计算

| 获取方式 | 指标    |
|------|-------|
| 打点数据 | 首帧时间  |
|      | 卡顿时间  |
|      | 丢帧率   |
|      | 音视频帧差异 |
|      | 视频基础数据 |
|      | ijk进程CPU占用 |
|      | ijk进程内存占用|
| cv计算 | 首帧时间  |
|      | 卡顿率   |
|      | 花屏率|
|      |黑屏率|
|      | 播放异常率 |

## 二 服务接口
### 1 打点数据获取
- 接口描述: 该接口用于获取打点指标, 实时返回
- Method: ** POST **
- URL: /player/index/dot
- Header:
- Body: 
```json5
{
  "buvid": "xxxx", // 用户id
  "device_id": "xxxxx" // 设备ID
  "video_start_time": "xxxxx" // 开始播放视频时间
  "video_end_time": "xxxx" // 结束播放时间
}
```

- Response:
- Body:
```json5
{
   "code": 0,
   "message": "Success",
   "index": {
        "video_base_info":{  // 视频基础信息
            "video_duration":"91463", // 视频总时长
            "audio_duration":"91463", // 音频总时长
            "video_bitrate":"244017", // 视频码率，音频码率
            "audio_bitrate":"67269" // 音频码率
        },
        "exit_error_info":{  // 视频播放item退出时候信息采集
            "last_audio_net_error_code":"-1005", // 最后一次音频网络错误， -1005表示正常
            "last_video_net_error_code":"-1005", // 最后一次视频网络错误
            "exit_player_status":"3" // 退出时候播放器状态，PLAYER_IDLE = 0, PLAYER_PREPAREING = 1, 
                                     // PLAYER_SWITCH_ITEM = 2, PLAYER_PLAY =3, PLAYER_PAUSE = 4, 
                                     // PLAYER_BUFFERING = 5, PLAYER_ERROR = 6， PLAYER_COMPLETE = 7
        },
        "first_video_time":"312", // 视频首帧时间
        "first_audio_time":"312", // 音频首帧时间
        "frame_loss_rate":"0.0",  // 丢帧率
        "freeze_times":"0",       // 卡顿加载次数(不包含seek)
        "freeze_rate":0.0277637440867129, // 卡顿加载率(包含seek后的)
        "buffering_total_time":1209,     // 加载总时长
        "asset_update_count":"0",        // 
        "audio_pts_diff_time":"0",       // 音频pts跟实际播放的偏差
        "ijk_cpu_rate":"15",             // ijk进程 cpu占用率
        "ijk_mem":"0"                    // ijk进程 内存占用率
    }
}
```
### 2 CV方式获取
#### 上传视频接口
- 接口描述: 该接口通过cv识别方式播放质量指标，异步接口, 上传视频文件接口
- Method: ** POST **
- URL: /player/video/upload
- Header:
```json5
{"Content-Type": "multipart/form-data"}
```
- Body: 
```json5
{
  "file":  [] // 视频文件
}
```
- example
```python
import requests

url = "http://0.0.0.0:2233/player/video/upload"

payload = {"index_types": ["FIRSTFRAME", "STARTAPP", "BLACKFRAME", "BLURREDFRAME", "FREEZEFRAME"]}
files = [
  ('file', open('/xxx/xxx/screen.mp4','rb'))
]
headers = {
  "Content-Type": "multipart/form-data"
}

response = requests.request("POST", url, headers=headers, data = payload, files = files)

print(response.text.encode('utf8'))

```
- Response:
- Body:
```json5
// 正常返回
{
  "code": 0,
  "msg": "update successfully",
  "task_id": "xxxxx"
},

// 输入参数错误
{
 "code": -1,
 "message": "input error"
}

```

#### 获取指标接口
- 接口描述: 该接口通过cv识别方式或者播放质量指标，获取指标接口
- Method: ** GET **
- URL: /player/index/cv?<task_id>
- Header:
- Body: 
```json

```

- Response:
- Body:
```json5
{
    "image_dict":{
        "0":[
            [
                "http://uat-i0.hdslb.com/bfs/davinci/9a5e3cbf551e45dd3d6620f6bb5ab12999f41c64.png",
                0.9565997217875153
            ],
            [
                "http://uat-i0.hdslb.com/bfs/davinci/2375def7e5c03b65ff86cdb8c8a660f700790508.png",
                1.1479196661450184
            ]
        ],
        "1":Array[2],
        "2":Array[55],
        "3":Array[4]
    },
    "first_frame_time":1.0434002782124847,
    "black_frame_list":Array[0],
    "freeze_frame_list":[
        Object{...},
        Object{...},
        Object{...},
        Object{...},
        {
            "freeze_start_time":"3.02808",
            "freeze_duration_time":"0.37",
            "freeze_end_time":"3.39808"
        }
    ]
}
```

## 三 安装服务

## 四 其他说明
不足与改进：
- 视频帧太多，不需要每一帧都输出，可以适当删除一些帧(解决)
- 分帧效率太低，一边分帧一边预测(解决)
- 预测服务效率低，调研tensorflow serving的效率(解决)
- 黑屏，花屏，卡顿预测暂时没有上线

docker打包：
- docker build  -t player-index-server:2.0 . --force-rm=true --rm=true
- docker tag player-index-server:2.0 hub.bilibili.co/luoyadong/player-index-server:2.0
- docker push hub.bilibili.co/luoyadong/player-index-server:2.0

依赖文档：
- https://kkroening.github.io/ffmpeg-python/












