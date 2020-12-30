# /usr/bin/env python
# -*- coding: utf-8 -*-

from flask import Flask
from app.controller import player_app, image_app

app = Flask(__name__)
app.register_blueprint(player_app, url_prefix='/player')
app.register_blueprint(image_app, url_prefix='/image')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=2233, debug=True, threaded=True)
