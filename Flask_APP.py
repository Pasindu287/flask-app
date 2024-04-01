from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
from flask_cors import CORS
import io


app = Flask(__name__)
CORS(app)

# Load the pre-trained model
model = tf.keras.models.load_model('model.h5')

def preprocess_image(image):
    """
    Pre-process the image: resize, convert to grayscale, and normalize.
    """
    # Resize the image to match the input shape expected by the model
    image = image.resize((224, 224))
    
    # Convert the image to grayscale
    image = image.convert('L')
    
    # Convert the image to a numpy array
    image_array = np.array(image)
    
    # Normalize the image array
    image_array = image_array / 255.0
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

def process_prediction(prediction):
    """
    Process the prediction and return a human-readable response.
    """
    # Assuming the model outputs probabilities for each class
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)
    confidence_threshold = 0.5  # Confidence threshold to determine valid classification

    # Mapping class indices to class names
    class_map = {0: 'Spider Veins', 1: 'Varicose Veins', 2: 'Normal Legs',3: 'Other'}

    if confidence >= confidence_threshold and predicted_class_index in class_map:
        predicted_class_name = class_map[predicted_class_index]
    else:
        predicted_class_name = "Not Detected" 
    response = {
        'predicted_class': predicted_class_name,
        'confidence': float(confidence)
    }
    return response

@app.route('/')
def home():
    return 'Welcome to the Image Classifier API! Use the /predict endpoint to classify images.'

@app.route('/predict', methods=['POST'])
def predict():
    app.logger.info("Predict route reached, method: %s", request.method)
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    image = Image.open(io.BytesIO(file.read()))
    image_array = preprocess_image(image)
    prediction = model.predict(image_array)
    response = process_prediction(prediction)
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
