#!/bin/sh
python3 main.py & redis-server & celery -A app.tasks worker -l INFO --autoscale=25,6