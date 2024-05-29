import os
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt 
import numpy as np 
import tensorflow as tf 
import pandas as pd
from pathlib import Path
from test import predict_image, new_model
import logging

# Set the directories for uploads and processed files within the static directory
UPLOAD_FOLDER = os.path.join('static', 'uploads')
PROCESSED_FOLDER = os.path.join('static', 'processed')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
    try:
        # Read CSV file
        treatments_df = pd.read_csv('plant_disease_treatments.csv')
        plant_conditions = treatments_df[['Plant Type', 'Condition']].values.tolist()
        headers = ['Plant Type', 'Condition']
        return render_template('upload.html', headers=headers, plant_conditions=plant_conditions)
    except Exception as e:
        logging.error("Error loading upload form: %s", e)
        return "Internal Server Error", 500

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
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
            processed_filename, treatment, confidence = predict_image(save_location, new_model, filename)
            return render_template('upload.html', filename=processed_filename, treatment=treatment, confidence=confidence)
        else:
            flash('Allowed image types are -> png, jpg, jpeg, gif')
            return redirect(request.url)
    except Exception as e:
        logging.error("Error uploading image: %s", e)
        return "Internal Server Error", 500

@app.route('/processed/<filename>')
def display_image(filename):
    try:
        return send_from_directory(app.config['PROCESSED_FOLDER'], filename)
    except Exception as e:
        logging.error("Error displaying image: %s", e)
        return "Internal Server Error", 500

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    app.run()
