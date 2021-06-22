# /usr/bin/env python
# -*- coding: utf-8 -*-

"""
图像视频指标接口路由
"""
import os
import time
import traceback

from celery.result import AsyncResult
from fastapi import APIRouter, File, UploadFile

from app.data import DotItem
from app.factory import FormatChecker, LogManager
from app.model import PlayerIndex, ImageIndex
from app.tasks import celery_app, cv_index_task

player_app = APIRouter()  # 视频类接口
image_app = APIRouter()  # 图像类接口

format_handler = FormatChecker()
logger = LogManager("server.log").logger


@player_app.post('/index/dot')
def get_dot_index(item: DotItem):
    input_json = item.dict()
    model_handler = PlayerIndex(dot_info_dict=input_json)
    success_flag, result_json = model_handler.get_dot_index()
    if success_flag:
        return {
            "code": 0,
            "message": "Success",
            "index": result_json
        }
    else:
        return {
            "code": -2,
            "message": "calculate index error, plz check log",
        }


@player_app.post("/video/upload")
async def file_upload(index_types: list, file: UploadFile = File(...)):
    try:
        res = await file.read()
        if not format_handler.video_index_cv_check(index_types, file):
            raise Exception("input error")
        else:
            base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'temp_dir')
            if not os.path.exists(base_path):
                os.mkdir(base_path)
            file_path = os.path.join(base_path, str(time.time()) + file.filename)
            # ffmpeg直接读FileStorage的方法暂时没有调研到，所以保存到一个临时文件
            with open(file_path, "wb") as f:
                f.write(res)
            logger.info(file_path)
            r = cv_index_task.delay({"temp_video_path": file_path, "index_types": index_types})
            task_id = r.task_id
            return {
                "code": 0,
                "message": "Success",
                "task_id": task_id
            }
    except Exception as e:
        return {"code": -1, "message": str(e)}


@player_app.get('/index/cv')
def get_cv_index(task_id: str):
    task_id = task_id
    model_handler = PlayerIndex()
    if task_id:
        res = AsyncResult(task_id, app=celery_app)
        if res.failed():
            return {
                "code": -2,
                "message": "task failed, plz try again"}
        elif res.status == "PENDING" or res.status == "RETRY" or res.status == "STARTED":
            history_result = model_handler.get_history_parsing_task(task_id)
            if history_result:
                return {
                    "code": 0,
                    "data": history_result,
                    "message": "Success"}
            else:
                return {
                    "code": -4,
                    "message": "plz wait a moment, task status is {}".format(res.status)}

        elif res.status == "SUCCESS":
            if res.result[0]:
                logger.info(res)
                model_handler.save_tasks_db({"task_id": task_id,
                                             "task_result": res.result[1]}
                                            )
                return {
                    "code": 0,
                    "data": res.result[1],
                    "message": "Success"}
            else:
                return {
                    "code": -3,
                    "message": "inner error"}
        else:
            return {
                "code": -3,
                "message": "inner error, return code error"}
    else:
        return {
            "code": -1,
            "message": "input error, make sure task id is correct"}


@player_app.post('/video/ssim')
async def get_ssim_index(file_src: UploadFile = File(...),
                         file_target: UploadFile = File(...)
                         ):
    try:
        res_src = await file_src.read()
        res_target = await file_target.read()

        base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'temp_dir')
        if not os.path.exists(base_path):
            os.mkdir(base_path)
        file_src_path = os.path.join(base_path, str(time.time()) + file_src.filename)
        file_target_path = os.path.join(base_path, str(time.time()) + file_target.filename)
        with open(file_src_path, "wb") as f:
            f.write(res_src)
        with open(file_target_path, 'wb') as f:
            f.write(res_target)
        model_handler = PlayerIndex(video_quality_dict={"src_video": file_src_path, "target_video": file_target_path})
        video_quality_result_index = model_handler.get_video_quality()
        os.remove(file_src_path)  # 删除临时视频文件
        os.remove(file_target_path)  # 删除临时固定帧率源
        return {
            "code": 0,
            "message": "Success",
            "data": video_quality_result_index
        }
    except Exception as err:
        return {"code": -1, "message": str(err)}


@player_app.post('/video/vmaf')
async def get_vmaf_score(file_input: UploadFile = File(...),
                         file_refer: UploadFile = File(...)
                         ):
    try:
        res_input = await file_input.read()  # 输入视频
        res_refer = await file_refer.read()  # 参考视频

        base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'temp_dir')
        if not os.path.exists(base_path):
            os.mkdir(base_path)
        file_input_path = os.path.join(base_path, str(time.time()) + file_input.filename)
        file_refer_path = os.path.join(base_path, str(time.time()) + file_refer.filename)
        with open(file_input_path, "wb") as f:
            f.write(res_input)
        with open(file_refer_path, 'wb') as f:
            f.write(res_refer)
        model_handler = PlayerIndex(video_quality_dict={"input_video": file_input_path, "refer_video": file_refer_path})
        video_quality_result_score = model_handler.get_video_quality_vmaf()
        os.remove(file_input_path)  # 删除临时视频文件
        os.remove(file_refer_path)
        if video_quality_result_score['vmaf_score'] is not None:
            return {
                "code": 0,
                "message": "Success",
                "data": video_quality_result_score
            }
        else:
            raise Exception("视频解析出现异常（可能由于视频分辨率不同）")
    except Exception as err:
        return {"code": -1, "message": str(err)}


@player_app.post('/video/colorcast-detect')
async def get_colour_cast_index(file_src: UploadFile = File(...)):
    """导入前视频偏色检测（无参考）"""
    if format_handler.api_video_colour_cast_detection_checker(file_src):
        try:
            file = await file_src.read()
            base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'temp_colour_cast_dir')
            if not os.path.exists(base_path):
                os.mkdir(base_path)
            file_path = os.path.join(base_path, 'video_' + str(time.time()) + file_src.filename)
            with open(file_path, "wb") as f:
                f.write(file)
            model_handler = PlayerIndex(colour_cast_dict={"video_path": file_path})
            colour_cast_result = model_handler.get_colour_cast_index()
            os.remove(file_path)
            return {
                "code": 0,
                "message": "Success",
                "data": colour_cast_result
            }
        except Exception as err:
            return {
                "code": -2,
                "message": str(err)
            }
    else:
        return {
            "code": -1,
            "message": "input error"
        }


@player_app.post('/video/reference-colorcast-detect')
async def get_colour_cast_with_reference_index(file_src: UploadFile = File(...),
                                               file_target: UploadFile = File(...)):
    """传入导入前后的两个视频偏色比较(有参考)
    """
    if format_handler.api_video_colour_cast_detection_checker(file_src) \
            and format_handler.api_video_colour_cast_detection_checker(file_target):
        try:
            res_src = await file_src.read()
            target_src = await file_target.read()
            base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'temp_colour_cast_with_reference_dir')
            if not os.path.exists(base_path):
                os.mkdir(base_path)
            file_src_path = os.path.join(base_path, 'video_src_' + str(time.time()) + file_src.filename)
            file_target_path = os.path.join(base_path, 'video_target_' + str(time.time()) + file_target.filename)
            with open(file_src_path, "wb") as f_src:
                f_src.write(res_src)
            with open(file_target_path, "wb") as f_target:
                f_target.write(target_src)
            if format_handler.api_video_colour_cast_content_checker(file_src_path, file_target_path):
                model_handler = PlayerIndex(colour_cast_dict={"src_video_path": file_src_path,
                                                              "target_video_path": file_target_path})
                colour_cast_result = model_handler.get_colour_cast_index_with_reference()
                os.remove(file_src_path)
                os.remove(file_target_path)
                return {
                    "code": 0,
                    "message": "Success",
                    "data": colour_cast_result
                }
            else:
                os.remove(file_src_path)
                os.remove(file_target_path)
                return {
                    "code": -3,
                    "message": "The input videos time are different"
                }
        except Exception as err:
            return {
                "code": -2,
                "message": str(err)
            }
    else:
        return {
            "code": -1,
            "message": "input error"
        }


@player_app.post('/index/silence')
async def get_silence_index(file_src: UploadFile = File(...)):
    if format_handler.silence_index_checker(file_src.filename):
        try:
            file = await file_src.read()
            base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'temp_dir')
            if not os.path.exists(base_path):
                os.mkdir(base_path)
            file_path = os.path.join(base_path, str(time.time()) + file_src.filename)
            with open(file_path, "wb") as f:
                f.write(file)
            model_handler = PlayerIndex(silence_info_dict={"video_path": file_path})
            silence_result_index = model_handler.get_silence_index()
            os.remove(file_path)
            return {
                "code": 0,
                "message": "Success",
                "data": silence_result_index
            }
        except Exception as err:
            return {
                "code": -2,
                "message": str(err)
            }
    else:
        return {
            "code": -1,
            "message": "input error"
        }


@image_app.post('/quality/black_white-detect')
async def judge_black_white_frame(file: UploadFile = File(...)):
    res_src = await file.read()
    if format_handler.api_image_checker(file.filename):
        image_handler = ImageIndex(res_src)
        result = image_handler.black_white_frame_detect()
        return {
            "code": 0,
            "message": "Success",
            "data": {"judge": result}
        }
    else:
        return {
            "code": -1,
            "message": "input error"}


@image_app.post('/quality/error-detect')
async def judge_error_frame():
    pass


@image_app.post('/quality/char-recognize')
async def frame_ocr(file: UploadFile = File(...)):
    res_src = await file.read()
    if format_handler.api_image_white_detection_checker(file):
        image_handler = ImageIndex(res_src)
        ocr_result_list = image_handler.frame_ocr()
        return {
            "code": 0,
            "message": "Success",
            "data": {"ocr_result": ocr_result_list}
        }
    else:
        return {
            "code": -1,
            "message": "input error"}


@image_app.post('/quality/blurred-detect')
async def blurred_frame_detect(file: UploadFile = File(...)):
    res_src = await file.read()
    if format_handler.api_image_white_detection_checker(file):
        image_handler = ImageIndex(res_src)
        result = image_handler.blurred_frame_detection()
        if result == -1:
            return {
                "code": -2,
                "message": "access model server error"
            }
        else:
            return {
                "code": 0,
                "message": "Success",
                "data": {"judge": result}
            }
    else:
        return {
            "code": -1,
            "message": "input error"}


@image_app.post('/quality/similarity')
async def caculate_similarity(file_src: UploadFile = File(...),
                                file_target: UploadFile = File(...)):
    res_src = await file_src.read()
    res_tar = await file_target.read()
    if format_handler.api_image_white_detection_checker(file_src) and \
            format_handler.api_image_white_detection_checker(file_target):
        image_handler = ImageIndex(res_src,target_file=res_tar)
        similarity_result = image_handler.caculate_similarity()
        return {
            "code": 0,
            "message": "Success",
            "data": {"feature_sim_score": similarity_result}
        }
    else:
        return {
            "code": -1,
            "message": "input error"}



@image_app.post('/quality/watermark-detect')
async def watermark_frame_detect(file: UploadFile = File(...)):
    res_src = await file.read()
    if format_handler.api_image_white_detection_checker(file):
        image_handler = ImageIndex(res_src)
        result = image_handler.watermark_frame_detection()
        if result == -1:
            return {
                "code": -2,
                "message": "access model server error"
            }
        else:
            return {
                "code": 0,
                "message": "Success",
                "data": {"judge": result}
            }
    else:
        return {
            "code": -1,
            "message": "input error"}


@image_app.post('/quality/horizontal-frame-detect')
async def horizontal_frame_detect(file: UploadFile = File(...)):
    res_src = await file.read()
    if format_handler.api_image_white_detection_checker(file):
        image_handler = ImageIndex(res_src)
        result = image_handler.frame_horizontal_portrait_detect()
        if result == -1:
            return {
                "code": -2,
                "message": "access model server error"
            }
        else:
            return {
                "code": 0,
                "message": "Success",
                "data": {"judge": result}
            }
    else:
        return {
            "code": -1,
            "message": "input error"}


@image_app.post('/quality/clarity-detect')
async def clarity_detect(file: UploadFile = File(...)):
    res_src = await file.read()
    if format_handler.api_image_checker(file.filename):
        image_handler = ImageIndex(res_src)
        result = image_handler.frame_clarity_detect()
        if result == -1:
            return {
                "code": -2,
                "message": "access model server error"
            }
        else:
            return {
                "code": 0,
                "message": "Success",
                "data": {"judge": result}
            }
    else:
        return {
            "code": -1,
            "message": "input error"}


@image_app.post('/quality/green-frame-detect')
async def green_frame_detect(file: UploadFile = File(...)):
    res_src = await file.read()
    if format_handler.api_image_checker(file.filename):
        image_handler = ImageIndex(res_src)
        result = image_handler.green_frame_detect()
        if result == -1:
            return {
                "code": -2,
                "message": "access model server error"
            }
        else:
            return {
                "code": 0,
                "message": "Success",
                "data": {"judge": result}
            }
    else:
        return {
            "code": -1,
            "message": "input error"}


@image_app.post('/quality/colorlayer-detect')
async def colorlayer_detect(file: UploadFile = File(...)):
    """颜色区域检测，分为红 绿 蓝
    """
    res_src = await file.read()
    if format_handler.api_image_white_detection_checker(file):
        image_handler = ImageIndex(res_src)
        result = image_handler.frame_colorlayer_detect()
        if result == -1:
            return {
                "code": -2,
                "message": "access model server error"
            }
        else:
            return {
                "code": 0,
                "message": "Success",
                "data": {"judge": result}
            }
    else:
        return {
            "code": -1,
            "message": "input error"}


@image_app.post('/quality/image-match')
async def get_image_match_res(file_src: UploadFile = File(...), file_target: UploadFile = File(...)):
    """ 图像匹配接口
    param: file_src: 模版图像
    param: file_target: 匹配图像
    """
    if format_handler.api_image_white_detection_checker(file_src) \
            and format_handler.api_image_white_detection_checker(file_target):
        try:
            res_src = await file_src.read()
            target_src = await file_target.read()
            image_handler = ImageIndex(quality_file=res_src, target_file=target_src)
            match_res_dict = image_handler.image_matching()
            return {
                "code": 0,
                "message": "Success",
                "data": match_res_dict
            }
        except Exception as err:
            traceback.print_exc()
            return {
                "code": -2,
                "message": str(err)
            }
    else:
        return {
            "code": -1,
            "message": "input error"
        }
