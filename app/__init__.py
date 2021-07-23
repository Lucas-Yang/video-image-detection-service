# -*- coding: utf-8 -*-
"""
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration

sentry_sdk.init(
    dsn="http://83fa40d91cbf40dab3f6ece81233bd6b@10.23.255.74:9000/7",
    integrations=[FlaskIntegration()],
    traces_sample_rate=1.0
)
"""
from paddleocr import PaddleOCR
import os
base_path = os.getcwd()
cls_path = base_path + '/model/.paddleocr/2.1/cls'
det_path = base_path + '/model/.paddleocr/2.1/det'
rec_path = base_path + '/model/.paddleocr/2.1/rec'
paddle_ocr = PaddleOCR(det_model_dir=det_path, rec_model_dir=rec_path, cls_model_dir=cls_path,
                       use_gpu=False, use_angle_cls=True, lang="ch")