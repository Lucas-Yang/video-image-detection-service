# /usr/bin/env python
# -*- coding: utf-8 -*-
import sentry_sdk
from flask import Flask
from sentry_sdk.integrations.flask import FlaskIntegration
from app.controller import player_app, image_app
sentry_sdk.init(
    dsn="http://83fa40d91cbf40dab3f6ece81233bd6b@10.23.255.74:9000/7",
    integrations=[FlaskIntegration()],
    traces_sample_rate=1.0
)

app = Flask(__name__)
app.register_blueprint(player_app, url_prefix='/player')
app.register_blueprint(image_app, url_prefix='/image')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=2233, debug=True, threaded=True)
