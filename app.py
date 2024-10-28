import mimetypes
import os
import io
import requests

# Désactiver les optimisations oneDNN
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Désactiver les GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Set the environment variable to use the TensorFlow 2.0 backend
os.environ["SM_FRAMEWORK"] = "tf.keras"

import segmentation_models as sm
sm.set_framework('tf.keras')
from segmentation_models import FPN

import tensorflow as tf
from tensorflow.keras.models import load_model
from segmentation_models.losses import categorical_crossentropy

from flask import Flask, request, jsonify, send_file
from PIL import Image

import numpy as np
from matplotlib import colors
import dill


# Path to the Keras model
model_path = "./model/model.keras"

# Load the Keras model
MODEL = load_model(model_path, custom_objects={'CategoricalCELoss':categorical_crossentropy}, compile=False)

# Input dimensions expected by your Keras model
MODEL_INPUT_WIDTH = 256
MODEL_INPUT_HEIGHT = 128

# Get the absolute path of the directory containing app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Create the 'generated' directory if it doesn't exist
GENERATED_DIR = os.path.join(BASE_DIR, 'generated')
os.makedirs(GENERATED_DIR, exist_ok=True)

def generate_img_from_mask(mask, colors_palette=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']):
    # Generate a color image array from a segmented mask
    # mask - numpy array of dimension
    # colors_palette - list - color to be assigned to each class

    id2category = {0: 'void',
                   1: 'flat',
                   2: 'construction',
                   3: 'object',
                   4: 'nature',
                   5: 'sky',
                   6: 'human',
                   7: 'vehicle'}

    # Start initializing the output image
    img_seg = np.zeros((mask.shape[0], mask.shape[1], 3), dtype='float')

    # Assign RGB channels
    for cat in id2category.keys():
        img_seg[:, :, 0] += mask[:, :, cat] * colors.to_rgb(colors_palette[cat])[0]
        img_seg[:, :, 1] += mask[:, :, cat] * colors.to_rgb(colors_palette[cat])[1]
        img_seg[:, :, 2] += mask[:, :, cat] * colors.to_rgb(colors_palette[cat])[2]

    return img_seg

def predict_segmentation(image_path):
    # Load and resize the image
    image = Image.open(image_path)
    image = image.resize((MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT))

    # Convert the image to a numpy array
    image_array = np.array(image)

    # Expand dimension of the image
    image_array = np.expand_dims(np.array(image_array), axis=0)
    
    # Predict the mask as an output of the model
    mask_predict = MODEL.predict(image_array)

    # Squeeze the first dimension of the mask.
    mask_predict = np.squeeze(mask_predict, axis=0)

    # Finally, generate a color image (RGB image) from the mask array
    mask_color = generate_img_from_mask(mask_predict) * 255

    return mask_color

app = Flask(__name__)







@app.route('/predict', methods=['POST'])
def predict():
      data = request.get_json()
      image_url = data.get('image_url')
      predicted_mask_filename = data.get('predicted_mask_filename')

      #### Replace the url in local mode #### Remove line when on azure server
      image_path = image_url.replace('http://localhost:8000/', '../frontend/static/')


      prediction = predict_segmentation(image_path)


      # Create the path to save the predicted mask
      predicted_mask_path = os.path.join(GENERATED_DIR, predicted_mask_filename)
      prediction_mask = Image.fromarray(prediction.astype(np.uint8))
      prediction_mask.save(predicted_mask_path)
      
      # Envoyer les informations à une autre route
      return jsonify({
          'message': 'Prediction completed successfully',
          'predicted_mask_path': predicted_mask_path
      }), 200
        

if __name__ == '__main__':
    app.run(debug=True, port=8001)