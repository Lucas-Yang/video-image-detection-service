import os
import sys

from paddleocr import PaddleOCR


def model_path():
    if hasattr(sys, 'frozen'):
        return os.path.dirname(sys.executable)
    return os.path.dirname(__file__)


cls_path = model_path() + '/.paddleocr/2.1/cls'
det_path = model_path() + '/.paddleocr/2.1/det'
rec_path = model_path() + '/.paddleocr/2.1/rec'
paddle_ocr_client = PaddleOCR(det_model_dir=det_path, rec_model_dir=rec_path, cls_model_dir=cls_path,
                       use_gpu=False, use_angle_cls=True, lang="ch")
