import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename

from app import app
from test import predict_image, new_model

# Configuration constants
UPLOAD_FOLDER = os.path.join('static', 'uploads')
PROCESSED_FOLDER = os.path.join('static', 'processed')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Flask application configuration
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_plant_conditions():
    try:
        # Read CSV file
        treatments_df = pd.read_csv('plant_disease_treatments.csv')
        plant_conditions = treatments_df[['Plant Type', 'Condition']].values.tolist()
        headers = ['Plant Type', 'Condition']
        return headers, plant_conditions
    except Exception as e:
        logging.error("Error loading plant conditions: %s", e)
        return [], []

@app.route('/')
def upload_form():
    headers, plant_conditions = get_plant_conditions()
    return render_template('upload.html', headers=headers, plant_conditions=plant_conditions)

@app.route('/upload', methods=['POST'])
def upload_image():
    headers, plant_conditions = get_plant_conditions()
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
            logging.debug(f"Confidence (raw): {confidence}")
            confidence_percentage = f"{confidence * 100:.2f}%"
            logging.debug(f"Confidence (percentage): {confidence_percentage}")
            return render_template('upload.html', filename=processed_filename, treatment=treatment, confidence=confidence_percentage, headers=headers, plant_conditions=plant_conditions)
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
