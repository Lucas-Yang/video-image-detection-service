"""
自定义数据类
"""
from pydantic import BaseModel
from typing import Optional
from enum import Enum
from fastapi import Query


class DotItem(BaseModel):
    """ 打点查询参数定义
    """
    device_id: Optional[str] = Query(None, min_length=3, max_length=50)
    buvid: Optional[str] = Query(None, min_length=3, max_length=50)
    start_time: Optional[str] = Query(None, min_length=3, max_length=50)
    end_time: Optional[str] = Query(None, min_length=3, max_length=50)


class VideoQualityItem(str, Enum):
    """ 异步视频质量检测参数定义
    """
    FIRSTFRAME = "FIRSTFRAME"
    STARTAPP = "STARTAPP"
    BLACKFRAME = "BLURREDFRAME"
    FREEZEFRAME = "FREEZEFRAME"
    STARTAPPYOUKU = "STARTAPPYOUKU"
    STARTAPPIXIGUA = "STARTAPPIXIGUA"
    STARTAPPTENCENT = "STARTAPPTENCENT"
    STARTAPPIQIYI = "STARTAPPIQIYI"
    STARTAPPDOUYIN = "STARTAPPDOUYIN"
    STARTAPPCOMIC = "STARTAPPCOMIC"
