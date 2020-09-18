#!/bin/sh
celery -A app.tasks worker -P gevent --concurrency=1000 --loglevel=info --logfile=./log/celery.log &
gunicorn -c config.py run:app;