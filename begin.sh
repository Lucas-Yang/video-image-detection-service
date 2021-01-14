#!/bin/sh
gunicorn -c config.py run:app & redis-server & celery -A app.tasks worker -l INFO --autoscale=25,6