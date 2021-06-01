from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, Docker!'


@app.route('/genmusic')
def gen_music():
    return 'Hello, Docker!'

