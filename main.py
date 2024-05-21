import os
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt 
import numpy as np 
import os 
import PIL 
import tensorflow as tf 
import glob
from pathlib import Path
import keras_tuner as kt
import random
import seaborn as sns
from PIL import Image
from typing import Dict, List
  
from tensorflow import keras
from tensorflow.keras import layers 
from tensorflow.keras.models import Sequential 

from test import predict_image
from test import new_model




ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		save_location = os.path.join(app.config['UPLOAD_FOLDER'], filename)
		file.save(save_location)
		flash('Image successfully uploaded and displayed below')
		processed_filename, treatment = predict_image(save_location, new_model, filename)
		# processed_location = os.path.join(app.config['PROCESSED_FOLDER'],filename)
		return render_template('upload.html', filename=processed_filename, treatment = treatment)
	else:
		flash('Allowed image types are -> png, jpg, jpeg, gif')
		return redirect(request.url)

@app.route('/processed/<filename>')
def display_image(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

# @app.route('/display/<filename>')
# def display_image(filename):
# 	#print('display_image filename: ' + filename)
# 	return redirect(url_for('static', filename='uploads/' + filename), code=301)


	
# def process_img(filename):
#     data_dir = "/static/archive/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
#     train_dir = Path(os.path.join(data_dir, "train"))
#     valid_dir = Path(os.path.join(data_dir, "valid"))
#     diseases = os.listdir(train_dir)
#     jpg_train_files = list(Path(train_dir).rglob('*/*.jpg'))
#     batch_size = 32
#     img_height = 256
#     img_width = 256
#     train_ds = tf.keras.utils.image_dataset_from_directory(
#         train_dir,
#         seed=123,
#         image_size=(img_height, img_width),
#         batch_size=batch_size)
#     # load the trained model
#     new_model = tf.keras.models.load_model('my_model.keras')
#     class_labels = train_ds.class_names
#     name_id_map = dict(zip(range(len(class_labels)), class_labels))
#     def predict_image(image_path, model):
#         # open and preprocess the image
#         img = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
#         img_array = tf.keras.utils.img_to_array(img)
#         img_array = tf.expand_dims(img_array, 0) # Create a batch

#         predictions = model.predict(img_array)
#         im_class = tf.argmax(predictions[0], axis=-1)
#         class_title = name_id_map.get(int(im_class))

#         # display the image with prediction and confidence
#         plt.imshow(np.squeeze(img_array.numpy().astype("uint8")))
#         plt.title(f"Class Predicted: {class_title}")
#         plt.axis("off")
#         plt.show()
#     predict_image(filename, new_model)
#     return render_template('upload.html', filename=filename)

if __name__ == "__main__":
    app.run()