"""
This script runs the AnalysisSystems application using a development server.
"""

from os import environ
from AnalysisSystems import app
import os

if __name__ == '__main__':
    HOST = environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(environ.get('SERVER_PORT', '5555'))
    except ValueError:
        PORT = 5555
    app.config['SESSION_TYPE'] = 'filesystem'
    app.config['SECRET_KEY'] = os.urandom(24)
    app.run(HOST, PORT)
