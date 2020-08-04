# 播放器播放指标服务

## 一 概述
本服务用于获取播放视频的质量指标。
后端用两种方式来计算播放指标：端上播放器打点数据获取，cv方式计算

| 获取方式 | 指标    |
|------|-------|
| 打点数据 | 首帧时间  |
|      | 卡顿时间  |
|      | 丢帧率   |
|      | 视频基础数据 |
|      | ijk进程占用系统资源 |
| cv计算 | 首帧时间  |
|      | 卡顿率   |
|      | 播放异常率 |

## 二 服务接口
### 1 打点数据获取
- 接口描述: 该接口用于获取打点指标, 实时返回
- Method: ** POST **
- URL: /v1/index/dot
- Header:
- Body: 
```json5
{
  "uid": "xxxx", // 用户id
  "bvid": "xxxx", // 设备id
  "video_id": "xxxx" // 播放视频id
}
```

- Response:
- Body:
```json5
{
   "code": 0,
   "message": "Success",
   "index": {
            "video_duration": "", // 视频总时长 
            "audio_duration": "", // 音频总时长
            "video_bitrate": "", // 视频码率
            "audio_bitrate": "", // 音频码率
            "first_video_time": "", // 视频首帧时间(包括渲染)
            "first_audio_time": "", // 音频首帧时间
            "freeze_rate": "", // 丢帧率
            "asset_update_count": "", // 资源刷新次数
            "audio_pts_diff_time": "", // 音频播放偏差 
            "ijk_cpu_rate": "", // ijk进程cpu占有率
            "ijk_mem": "" //ijk进程内存占用
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

payload = {}
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
```json
{
  "code": 0,
  "msg": "update successfully",
  "task_id": "xxxxx"
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
  "first_video_time": "",
  "black_screen_rate": "",
  "freeze_rate": "",
  "error_rate": ""
}
```

## 三 安装服务

## 四 其他说明











