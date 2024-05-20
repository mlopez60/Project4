# Importing libraries 
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


data_dir = "/Users/mlopez/Desktop/archive/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
train_dir = Path(os.path.join(data_dir, "train"))
valid_dir = Path(os.path.join(data_dir, "valid"))
diseases = os.listdir(train_dir)
jpg_train_files = list(Path(train_dir).rglob('*/*.jpg'))
len(jpg_train_files) 


batch_size = 32
img_height = 256
img_width = 256

train_ds = tf.keras.utils.image_dataset_from_directory(
  train_dir,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


# load the trained model
new_model = tf.keras.models.load_model('my_model.keras')

# define class labels
class_labels = train_ds.class_names

# Show the model architecture
new_model.summary()


name_id_map = dict(zip(range(len(class_labels)), class_labels))
print(name_id_map)

# prediction function on single image
def predict_image(image_path, model):
    plt.switch_backend('Agg') 
    # open and preprocess the image
    img = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    im_class = tf.argmax(predictions[0], axis=-1)
    class_title = name_id_map.get(int(im_class))

    # display the image with prediction and confidence
    plt.imshow(np.squeeze(img_array.numpy().astype("uint8")))
    plt.title(f"Class Predicted: {class_title}")
    plt.axis("off")
    plt.savefig("ProcessedImage.png")