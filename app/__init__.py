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
paddle_ocr = PaddleOCR(use_gpu=False, use_angle_cls=True, lang="ch")