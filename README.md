# Image Segmentation API

This project provides a REST API for image segmentation, based on a deep learning model using the FPN architecture combined with EfficientNetB1. It takes an input image and returns a colorized segmentation mask corresponding to the detected classes.

## Model

- **Architecture**: FPN (Feature Pyramid Network)
- **Backbone**: EfficientNetB1
- **Loss function**: `cce_jaccard_loss`
- **Augmentation**: using personalized `imgaug` settings with 0.75 probability
- **Performance**: Score **IoU = 0.71**

The model is trained to segment images into 8 classes:
- `void`, `flat`, `construction`, `object`, `nature`, `sky`, `human`, `vehicle`

## Technologies Used

- Python 3.8+
- Flask
- TensorFlow / Keras
- segmentation-models
- Pillow
- NumPy
- Matplotlib
- imgaug


## Installation

- Clone the repository
- Install the required dependencies:
pip install -r requirements.txt


## Usage
To run the API backend, use the command:
flask run backend/app.py


## Contributing
As this is a school project, we are not currently accepting contributions.


## License
This project is for educational purposes and is not licensed.
