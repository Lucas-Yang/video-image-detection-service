#!/bin/sh
gunicorn -c config.py run:app --timeout 500 & celery -A app.tasks worker -l INFO --autoscale=20,6