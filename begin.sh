#!/bin/sh
gunicorn -c config.py run:app --timeout 500 & redis-server & celery -A app.tasks worker -l INFO --autoscale=25,6