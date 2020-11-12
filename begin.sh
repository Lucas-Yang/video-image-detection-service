#!/bin/sh
gunicorn -c config.py run:app & celery -A app.tasks worker -l INFO --autoscale=30,6