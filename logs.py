from flask import Flask
from src.logger import logging

app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def index():
    logging.info("We are testing our logging file")

    return "Welcome to my new project "

if __name__ == "__main__":
    app.run(debug = True) 