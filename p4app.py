
from flask import Flask, render_template, request
# import sqlite3
import csv
from PIL import Image
import numpy as np

UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.secret_key ="secret key"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route('/', methods = ['GET', 'POST'])
def homepage():
    if request.method == 'GET':
        return render_template('home.html', msg='')

    image = request.files['file']
    img = Image.open(image)
    img = np.array(img)

    print(img)
    print(img.shape)

    return render_template('home.html', msg='Your image has been uploaded')

# @app.route('/backend')
# def backend():
#     return render_template('backend.html')

# @app.route('/visuals')
# def index():
#     return render_template('index.html')

# @app.route('/data')

if __name__ == '__main__':
    app.run(debug=True)
