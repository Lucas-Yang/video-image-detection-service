# /usr/bin/env python
# -*- coding: utf-8 -*-

from flask import Flask, current_app
from app.controller import player_app
from app.factory import LogManager

app = Flask(__name__)
app.register_blueprint(player_app, url_prefix='/player')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=2233, debug=True)
