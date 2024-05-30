# Importing libraries
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from PIL import Image
from werkzeug.utils import secure_filename
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from app import app

# Define directories
data_dir = r"C:\\Users\\steve\\Desktop\\UCF-VIRT-DATA-PT-12-2023-U-LOLC\\Project 4\\Project4\\New Plant Diseases Dataset(Augmented)"
train_dir = Path(os.path.join(data_dir, "train"))
valid_dir = Path(os.path.join(data_dir, "valid"))

# List of diseases
diseases = os.listdir(train_dir)
jpg_train_files = list(Path(train_dir).rglob('*/*.jpg'))
print(f"Number of training images: {len(jpg_train_files)}")

# Image parameters
batch_size = 32
img_height = 256
img_width = 256

# Create training dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Load the trained model
new_model = tf.keras.models.load_model('my_model.keras')

# Define class labels
class_labels = train_ds.class_names

# Show the model architecture
new_model.summary()

# Create a map of class labels
name_id_map = dict(zip(range(len(class_labels)), class_labels))
print(name_id_map)

# Load treatment data
treatments_df = pd.read_csv('plant_disease_treatments.csv')
treatments_df.insert(0, 'New_ID', range(len(treatments_df)))

# Prediction function on a single image
def predict_image(image_path: str, model, original_filename: str):
    plt.switch_backend('Agg')
    img = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    probabilities = tf.nn.softmax(predictions[0])  # Convert logits to probabilities
    confidence = np.max(probabilities)  # Get the highest probability

    print(f"\nPredictions: {predictions[0]}\nSum: {sum(predictions[0])}\n")
    print(f"\nProbabilities: {probabilities}\nSum: {sum(probabilities)}\n")

    im_class = tf.argmax(predictions[0], axis=-1)
    class_title = name_id_map.get(int(im_class))
    treatment = treatments_df.loc[treatments_df.New_ID.eq(int(im_class)), 'Treatment'].values[0]

    plt.imshow(np.squeeze(img_array.numpy().astype("uint8")))
    plt.title(f"Class Predicted: {class_title}\nConfidence: {confidence:.2f}")
    plt.axis("off")
    processed_filename = secure_filename(original_filename)
    output_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
    plt.savefig(output_path)

    print(treatment)
    return processed_filename, treatment, confidence
