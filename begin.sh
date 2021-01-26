#!/bin/sh
python main.py & redis-server & celery -A app.tasks worker -l INFO --autoscale=25,6