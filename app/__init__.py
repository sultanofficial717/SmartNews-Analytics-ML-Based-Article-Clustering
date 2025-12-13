"""Initialize Flask app package"""

from flask import Flask

def create_app(config_name='development'):
    """Application factory"""
    from app import app as flask_app
    return flask_app
