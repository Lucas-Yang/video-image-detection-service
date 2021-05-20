# /usr/bin/env python
# -*- coding: utf-8 -*-

"""
fastapi 接口启动
"""
from app.controller_fastapi import player_app, image_app
from fastapi import FastAPI
import uvicorn

app = FastAPI()
app.include_router(player_app, prefix="/player")
app.include_router(image_app, prefix="/image")

if __name__ == '__main__':
    uvicorn.run('main:app', host="0.0.0.0", port=8090, reload=True, log_level="debug")
