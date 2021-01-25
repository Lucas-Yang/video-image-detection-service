# 视频质量&播放器指标 服务

## 一 概述
本服务用于获取 视频质量指标，图像指标，播放器播放质量指标。


| 指标类型 | 指标    |
|------|-------|
|播放器打点指标 | 首帧时间  |
|      | 卡顿时间  |
|      | 丢帧率   |
|      | 音视频帧差异 |
|      | 视频基础数据 |
|      | ijk进程CPU占用 |
|      | ijk进程内存占用|
| 视频显式指标 | 粉版播放器首帧时间  |
|      | 粉版app启动时间(双端)  |
|      | 爱奇艺启动时间(双端)  |
|      | 腾讯视频启动时间(双端)  |
|      | 优酷启动时间(双端)  |
|      | 西瓜视频启动时间(双端)   |
|      | 抖音启动时间(双端)   |
|      | 卡顿时间段列表  |
|      | 花屏时间戳列表(暂未上线)|
|      | 黑屏时间段列表|
|      | 音画不同步检测(暂未上线)|
| 视频质量指标|全参考视频质量指标： SSIM|
|      |全参考视频质量指标： Vmaf(暂没上线)|
|      |全参考视频质量指标： PSNR(暂没上线)|
| 图像指标|蓝屏，黑屏，绿屏检测|
|       |bilibili错误检测(暂未上线)|
|       |花屏检测(暂未上线)|
|       |图像相似度检测(暂未上线)|
|       |图像OCR(暂未上线)|

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
            "audio_bitrate":"67269", // 音频码率
            "video_read_bytes": "xxxxx", //视频读取的总字节数  
            "audio_read_bytes": "xxxxx" // 音频读取的总字节数
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
指标参数：


```python
import requests

url = "http://0.0.0.0:2233/player/video/upload"

payload = {"index_types": ["FIRSTFRAME", "STARTAPP", "BLACKFRAME", "BLURREDFRAME", "FREEZEFRAME"]}
files = [
  ('file', open('/xxx/xxx/screen.mp4','rb'))
]

response = requests.request("POST", url, data = payload, files = files)

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
- 黑屏，花屏，卡顿预测暂时没有上线(卡顿 黑屏以及上线)

docker打包：
- docker build  -t player-index-server:2.0 . --force-rm=true --rm=true
- docker tag player-index-server:2.0 hub.bilibili.co/luoyadong/player-index-server:2.0
- docker push hub.bilibili.co/luoyadong/player-index-server:2.0

依赖文档：
- https://kkroening.github.io/ffmpeg-python/

## 五 升级记录

| 升级内容 | 升级时间    |
|------|-------|
|任务队列BACKEND替换成mongodb，方便任务数据持久化 |  1-21 |
|访问模型服务替换成长链接，新增连接池管理| 1-22|
|修改接口架构，替换flask为FASTAPI|1-25|












