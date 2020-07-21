# 播放器播放指标服务

## 一 概述
本服务用于获取播放视频的质量指标。
后端用两种方式来计算播放指标：端上播放器打点数据获取，cv方式计算

| 获取方式 | 指标    |
|------|-------|
| 打点数据 | 首帧时间  |
|      | 卡顿率   |
|      | 丢帧率   |
|      | 播放异常率 |
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
```json
{
  "uid": "xxxx", // 用户id
  "bvid": "xxxx", // 设备id
  "video_id": "xxxx" // 播放视频id
}
```

- Response:
- Body:
```json
{
  "first_video_time": "",
  "black_screen_rate": "",
  "freeze_rate": "",
  "error_rate": ""
}
```
### 2 CV方式获取
#### 上传视频接口
- 接口描述: 该接口通过cv识别方式播放质量指标，异步接口, 上传视频文件接口
- Method: ** POST **
- URL: /v1/index/cv
- Header:
- Body: 
```json
{
  "video_path": "xxxx" // 上传文件名
}
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
- URL: /v1/index/cv?<task_id>
- Header:
- Body: 
```json

```

- Response:
- Body:
```json
{
  "first_video_time": "",
  "black_screen_rate": "",
  "freeze_rate": "",
  "error_rate": ""
}
```

## 三 安装服务

## 四 其他说明











