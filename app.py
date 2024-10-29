from flask import Flask, request, jsonify
import os
from PIL import Image
import numpy as np
from matplotlib import colors
from io import BytesIO
import base64

# Set the environment variable to use the TensorFlow 2.0 backend
os.environ["SM_FRAMEWORK"] = "tf.keras"

import segmentation_models as sm
sm.set_framework('tf.keras')
import tensorflow as tf
from tensorflow.keras.models import load_model
from segmentation_models.losses import categorical_crossentropy


# --------------------------------------------------------------------
# VARIABLES
# --------------------------------------------------------------------

# Get the absolute path of the directory containing app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_API_URL = 'https://projet8frontend-ejangbejgzeuapcv.westeurope-01.azurewebsites.net'
FRONTEND_IMAGES_DIR = os.path.join(FRONTEND_API_URL, 'static/images/source')


# Path to the Keras model
model_path = "./model/model.keras"

# Load the Keras model
MODEL = load_model(model_path, custom_objects={'CategoricalCELoss':categorical_crossentropy}, compile=False)

# Input dimensions expected by your Keras model
MODEL_INPUT_WIDTH = 256
MODEL_INPUT_HEIGHT = 128


# --------------------------------------------------------------------
# FUNCTIONS
# --------------------------------------------------------------------

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


# --------------------------------------------------------------------
# APP
# --------------------------------------------------------------------

app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():
    print('--- request :', request)
    data = request.get_json()
    predicted_mask_filename = data.get('predicted_mask_filename')
    image_name = data.get('image_name')
    print('--- image_name :', image_name)
    
    # Create the path to the image
    image_path = os.path.join(FRONTEND_IMAGES_DIR, image_name)

    print('--- image path :', image_path)
    # Make prediction
    prediction = predict_segmentation(image_path)

    # Transform the image to send to frontend api
    mask_image = Image.fromarray(prediction.astype(np.uint8))
    # Save the mask image to a BytesIO object
    img_io = BytesIO()
    mask_image.save(img_io, 'PNG')
    img_io.seek(0)
    
    img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
    print('--- img_base64 :', img_base64)
    # Return a response
    return jsonify({
        'message': 'Prediction completed successfully',
        'predicted_mask': img_base64
    }), 200
        

if __name__ == '__main__':
    app.run(debug=True, port=8001)