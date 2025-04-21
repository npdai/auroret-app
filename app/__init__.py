from flask import Flask # type: ignore


def create_app():
    app = Flask(__name__)
    app.secret_key = 'your_secret_key'  # You can use a random string here

    from .routes import main
    app.register_blueprint(main)

    return app

