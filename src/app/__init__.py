# src/app/__init__.py

from flask import Flask
import os

def create_app():
    app = Flask(__name__,
                template_folder="templates",
                static_folder="static")
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-key")

    # absolute import of routes blueprint
    from app.routes import bp as routes_bp
    app.register_blueprint(routes_bp)
    return app
