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
|      | ijk进程内存占用 |
|      | 异步接口 |
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
|      | 静音检测(初版上线)| 
|      | 色差检测(初版上线)|
|      | 相似度检测(初版上线)|
| 视频质量指标|全参考视频质量指标： SSIM|
|      |全参考视频质量指标： Vmaf|
|      |全参考视频质量指标： PSNR|
|      |无参考视频质量指标：NIQE|
| 图像指标|蓝屏，黑屏，绿屏检测|
|       |bilibili错误检测(暂未上线)|
|       |花屏检测(轻视频版本上线，其他需求待适配开发)|
|       |图像模板匹配(初版上线)|
|       |图像OCR(初版上线)|
|       |黑屏 白屏检测(初版上线)|
|       |过度渲染检测(初版上线)|
|       |横竖屏检测-运营中心(初版上线)|
|       |清晰度检测-运营中心(初版上线)|
|       |水印检测-运营中心(初版上线)|

## 二 服务接口

### 1 打点数据获取

- 接口描述: 该接口用于获取打点指标, 实时返回
- Method: ** POST **
- URL: /player/index/dot
- Header:
- Body:

```json5
{
  "buvid": "xxxx",
  // 用户id
  "device_id": "xxxxx"
  // 设备ID
  "video_start_time": "xxxxx"
  // 开始播放视频时间
  "video_end_time": "xxxx"
  // 结束播放时间
}
```

- Response:
- Body:

```json5
{
  "code": 0,
  "message": "Success",
  "index": {
    "video_base_info": {
      // 视频基础信息
      "video_duration": "91463",
      // 视频总时长
      "audio_duration": "91463",
      // 音频总时长
      "video_bitrate": "244017",
      // 视频码率，音频码率
      "audio_bitrate": "67269",
      // 音频码率
      "video_read_bytes": "xxxxx",
      //视频读取的总字节数  
      "audio_read_bytes": "xxxxx"
      // 音频读取的总字节数
    },
    "exit_error_info": {
      // 视频播放item退出时候信息采集
      "last_audio_net_error_code": "-1005",
      // 最后一次音频网络错误， -1005表示正常
      "last_video_net_error_code": "-1005",
      // 最后一次视频网络错误
      "exit_player_status": "3"
      // 退出时候播放器状态，PLAYER_IDLE = 0, PLAYER_PREPAREING = 1, 
      // PLAYER_SWITCH_ITEM = 2, PLAYER_PLAY =3, PLAYER_PAUSE = 4, 
      // PLAYER_BUFFERING = 5, PLAYER_ERROR = 6， PLAYER_COMPLETE = 7
    },
    "first_video_time": "312",
    // 视频首帧时间
    "first_audio_time": "312",
    // 音频首帧时间
    "frame_loss_rate": "0.0",
    // 丢帧率
    "freeze_times": "0",
    // 卡顿加载次数(不包含seek)
    "freeze_rate": 0.0277637440867129,
    // 卡顿加载率(包含seek后的)
    "buffering_total_time": 1209,
    // 加载总时长
    "asset_update_count": "0",
    // 
    "audio_pts_diff_time": "0",
    // 音频pts跟实际播放的偏差
    "ijk_cpu_rate": "15",
    // ijk进程 cpu占用率
    "ijk_mem": "0"
    // ijk进程 内存占用率
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
{
  "Content-Type": "multipart/form-data"
}
```

- Body:

```json5
{
  "file": []
  // 视频文件
}
```

- example 指标参数：

```python
import requests

url = "http://0.0.0.0:2233/player/video/upload"

payload = {"index_types": ["FIRSTFRAME", "STARTAPP", "BLACKFRAME", "BLURREDFRAME", "FREEZEFRAME"]}
files = [
    ('file', open('/xxx/xxx/screen.mp4', 'rb'))
]

response = requests.request("POST", url, data=payload, files=files)

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
  "image_dict": {
    "0": [
      [
        "http://uat-i0.hdslb.com/bfs/davinci/9a5e3cbf551e45dd3d6620f6bb5ab12999f41c64.png",
        0.9565997217875153
      ],
      [
        "http://uat-i0.hdslb.com/bfs/davinci/2375def7e5c03b65ff86cdb8c8a660f700790508.png",
        1.1479196661450184
      ]
    ],
    "1": Array[
  2
],
"2": Array[55],
"3": Array[4]
},
"first_frame_time": 1.0434002782124847,
"black_frame_list": Array[0],
"freeze_frame_list":[
Object{...},
Object{...},
Object{...},
Object{...},
{
"freeze_start_time": "3.02808",
"freeze_duration_time": "0.37",
"freeze_end_time": "3.39808"
}
]
}
```

### 3 视频静音检测

- 接口描述：该接口用来检测视频是否为静音，返回结果中包括silence_start(静音开始时间)、silence_end(静音结束时间)、silence_duration(该段静音持续时间)。

- example 指标参数：

```python
import requests

url = "http://127.0.0.1:8090/player/index/silence"

files = {"file": open(filepath, "rb")}

response = requests.request("POST", url, files=files)

print(response.text.encode('utf8'))

```

- Response:

```json5
{
  "code": 0,
  "message": "Success",
  "data": {
    "silence_timestamps": [
      {
        "silence_start": 0,
        "silence_end": 87.98,
        "silence_duration": 87.98
      }
    ]
  }
}
```

### 4 视频质量评估VMAF

- 接口描述：该接口用来获取输入视频与参考视频的质量评估，默认采用的是 v0.6.1模型，返回结果中的vmaf_score为对应的分数，该分数使用 1080p
  显示屏，观看距离为3H。观看者对视频质量的评分为“差”，“一般”，“好”和“优秀”，粗略估计，可以认为：0～40：差；40～70：一般；70～85：好；85～100：优秀。

- example 指标参数：

```python
import requests

url = "http://127.0.0.1:8090/player/video/vmaf"

files = [
    ('file_input', open(video_file_path, 'rb')),
    ('file_refer', open(video_refer_path, 'rb'))
]

response = requests.request("POST", url, files=files)

print(response.text.encode('utf8'))

```

- Response:

```json5
{
  "code": 0,
  "message": "Success",
  "data": {
    "vmaf_score": "79.948197"
  }
}
```

### 5 图像清晰度检测

- 接口描述：该接口用来检测图像清晰度，默认采用的是NRSS算法，返回结果中的judge为对应的分数， 粗略估计，可以认为：0～0.05：不清晰；0.05~0.07：较不清晰；0.07~0.08：较清晰；0.08～1：清晰。

- example 指标参数：

```python
import requests

url = "http://127.0.0.1:8090/image/quality/clarity-detect"

files = {"file": open(filepath, "rb")}

response = requests.request("POST", url, files=files)

print(response.text.encode('utf8'))

```

- Response:

```json5
{
  "code": 0,
  "message": "Success",
  "data": {
    "judge": 0.22282582851089128
  }
}
```

### 6 图像绿屏检测

- 接口描述：该接口用来检测图像中像素的色相为绿的占比，返回结果中的green_ratio为对应的占比，details中H_value和count分别表示对应的色相号和该色相的像素个数。

- example 指标参数：

```python
import requests

url = "http://127.0.0.1:8090/image/quality/green-frame-detect"

files = {"file": open(filepath, "rb")}

response = requests.request("POST", url, files=files)

print(response.text.encode('utf8'))

```

- Response:

```json5
{
  "code": 0,
  "message": "Success",
  "data": {
    "judge": {
      "green_ratio": 1.0,
      "details": [
        {
          "H_value": 93,
          "count": 438152
        }
      ]
    }
  }
}
```

### 7 图像水印检测

- 接口描述：该接口用来检测图像是否存在水印及水印的类别，采用Yolov3算法和tensorflow-serving用于部署。 返回结果是否存在水印：否（空列表）；是（抖音 快手 好看 小红书）四种当前版本的水印类型。

- example 指标参数：

```python
import requests

url = "http://localhost:8090/image/quality/watermark-detect"

files = {"file": open(filepath, "rb")}

response = requests.request("POST", url, files=files)

print(response.text.encode('utf8'))

```

- Response:

```json5
{
  "code": 0,
  "message": "Success",
  "data": {
    "judge": [
      "好看"
    ]
  }
}
```

```json5
{
  "code": 0,
  "message": "Success",
  "data": {
    "judge": []
  }
}
```

### 8 过度渲染检测

- 接口描述：该接口用来检测图像被渲染了几次，是否在感官上使人不适。 分为绿色（渲染一次），蓝色（渲染两次），浅红色（渲染三次），深红色（渲染四次）。 其中绿色 蓝色 浅红目前可接受，深红色需要检测出来后进行修改。 返回结果为绿色 蓝色
  浅红色图层的比例（具体数值），深红色为是否存在。

- example 指标参数：

```python
import requests

url = "http://localhost:8090/image/quality/colorlayer-detect"

files = {"file": open(filepath, "rb")}

response = requests.request("POST", url, files=files)

print(response.text.encode('utf8'))

```

- Response:

```json5
{
  "code": 0,
  "message": "Success",
  "data": {
    "judge": {
      "blue": "36.50%",
      "green": "32.42%",
      "red": "30.55%",
      "exist_deep_red": true
    }
  }
}
```

### 9 黑白屏检测

- 接口描述：该接口用来检测图像是否为黑白屏。返回结果为黑色 白色像素在图像中所占的比值。

- example 指标参数：

```python
import requests

url = "http://localhost:8090/image/quality/black_white-detect"

files = {"file": open(filepath, "rb")}

response = requests.request("POST", url, files=files)

print(response.text.encode('utf8'))

```

- Response:

```json5
{
  "code": 0,
  "message": "Success",
  "data": {
    "judge": {
      "black_ratio": 0.9997608481262328,
      "white_ratio": 0.0
    }
  }
}
```

### 10 色差检测

- 接口描述：该接口用来检测图像是否发生色差。由于机器的色域不同，同一视频在不同机器上可能会发生偏色，目前色域为BT2020的视频在上传后会发生偏色。返回结果为该视频上传后是否会偏色及色域类型。

- example 指标参数：

```python
import requests

url = "http://localhost:8090/player/video/colorcast-detect"

files = {"file_src": open(filepath, "rb")}

response = requests.request("POST", url, files=files)

print(response.text.encode('utf8'))

```

- Response:

```json5
{
  "code": 0,
  "message": "Success",
  "data": {
    "judge": true,
    "color_primaries": "bt2020"
  }
}
```

### 11 图像花屏检测

- 接口描述：该接口用来检测图像是否为花屏。返回结果是或否。

- example 指标参数：

```python
import requests

url = "http://127.0.0.1:8090/image/quality/blurred-detect"

files = {"file": open(filepath, "rb")}

response = requests.request("POST", url, files=files)

print(response.text.encode('utf8'))

```

- Response:

```json5
{
  "code": 0,
  "message": "Success",
  "data": {
    "judge": true
  }
}
```

### 12 图像相似度检测

- 接口描述：该接口用来检测视频或图像的相似度。返回结果为相似度的数值。

- example 指标参数：

```python
import requests

url = "http://hassan.bilibili.co/image/quality/similarity-v2"

files = {"file_src": open(filepath_src, "rb"), "file_target": open(filepath_target, "rb")}

response = requests.request("POST", url, files=files)

print(response.text.encode('utf8'))

```

- Response:

```json5
{
  "code": 0,
  "message": "Success",
  "data": {
    "ssim_score": 1.0
  }
}
```

### 13 图像匹配

- 接口描述：该接口用来检测模板图像和目标图像的匹配。结果返回模板图像在目标图像中的中心坐标（匹配成功）或空列表（匹配失败）

- example 指标参数：

```python
import requests

url = "http://localhost:8090/image/quality/image-match"

files = {"file_src": open(filepath_src, "rb"), "file_target": open(filepath_target, "rb")}

response = requests.request("POST", url, files=files)

print(response.text.encode('utf8'))

```

### 13 图像OCR

- 接口描述：该接口用来检测图像中的文字。结果返回图像中检测到的文字。

- example 指标参数：

```python
import requests

url = "http://localhost:8090/image/quality/image-match"

files = {"file": open(filepath, "rb")}

response = requests.request("POST", url, files=files)

print(response.text.encode('utf8'))

```

- Response:

```json5
{
  "code": 0,
  "message": "Success",
  "data": {
    "ocr_result": [
      "园与"
    ]
  }
}
```
### 14 视频质量评估PSNR

- 接口描述：该接口用来获取输入视频与参考视频的质量评估，是一种全参考的客观评价方法，PSNR分数越高表示压缩后失真越小。可以认为0～20视频质量非常差，20～30视频质量差，30～40视频质量可以接受，>40视频质量好，接近原始视频。

- example 指标参数：

```python
import requests

url = "http://127.0.0.1:8090/player/video/psnr"

files = [
    ('file_input', open(video_file_path, 'rb')),
    ('file_refer', open(video_refer_path, 'rb'))
]

response = requests.request("POST", url, files=files)

print(response.text.encode('utf8'))

```

- Response:

```json5
{
  "code": 0,
  "message": "Success",
  "data": {
    "psnr_score": "27.684495"
  }
}
```
### 15 视频质量评估NIQE

- 接口描述：该接口用来获取输入视频的质量，是一种无参考的客观评价方法，NIQE分数越高表示视频质量越差。NIQE是用来衡量一副图像质量的，本项目中是对视频的每一帧都进行打分然后取平均值作为视频的最终得分。

- example 指标参数：

```python
import requests

url = "http://127.0.0.1:8090/player/video/niqe"

files = [
    ('file_input', open(video_file_path, 'rb'))
]

response = requests.request("POST", url, files=files)

print(response.text.encode('utf8'))

```

- Response:

```json5
{
  "code": 0,
  "message": "Success",
  "data": {
    "niqe_score": "26.2006"
  }
}
```


## 三 安装服务

## 四 其他说明

不足与改进：

- 视频帧太多，不需要每一帧都输出，可以适当删除一些帧(解决)
- 分帧效率太低，一边分帧一边预测(解决)
- 预测服务效率低，调研tensorflow serving的效率(解决)

docker打包：

- docker build -t player-index-server:2.0 . --force-rm=true --rm=true
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
|访问模型服务由http迁移到grpc| 待定|












