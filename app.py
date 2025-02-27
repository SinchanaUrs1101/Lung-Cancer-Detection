import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import logging

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Ensure uploads folder exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Logging setup
logging.basicConfig(level=logging.DEBUG)  # Change to DEBUG to get more detailed logs
logger = logging.getLogger(__name__)

# Load the trained model
MODEL_PATH = os.path.join(os.getcwd(), 'lung_cancer_model.keras')  # Correct model path and extension
try:
    model = load_model(MODEL_PATH)
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise RuntimeError("Model loading failed. Check the path and ensure the model exists.")

# Labels for prediction output
labels = ['Benign', 'Malignant']  # Model's output classes

# Default route for serving the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint to process uploaded image
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            logger.error("No file provided in request.")
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            logger.error("No file selected.")
            return jsonify({'error': 'No file selected'}), 400

        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        logger.info(f"File saved to {file_path}")

        # Preprocess the image
        img = load_img(file_path, target_size=(224, 224))  # Resize to 224x224, matching the training size
        img_array = img_to_array(img) / 255.0  # Normalize the image to [0,1]
        img_array = img_array.reshape((1, 224, 224, 3))  # Reshape to match model input (batch size, height, width, channels)

        # Log the preprocessed image shape for debugging
        logger.info(f"Processed image shape: {img_array.shape}")

        # Predict using the model
        prediction = model.predict(img_array)
        logger.info(f"Raw model prediction: {prediction}")

        # Assuming the model outputs a single probability for Malignant (using sigmoid activation)
        malignant_probability = float(prediction[0][0])  # Convert to standard Python float
        benign_probability = 1 - malignant_probability  # Benign is just the complement of Malignant

        # Log the probabilities for debugging
        logger.info(f"Malignant Probability: {malignant_probability}")
        logger.info(f"Benign Probability: {benign_probability}")

        # Adjust the threshold to classify as Malignant or Benign
        threshold = 0.9 # You can adjust this threshold to a higher value if needed

        # Prediction logic based on threshold
        if malignant_probability > threshold:
            predicted_label = labels[1]  # Malignant if the probability exceeds threshold
        else:
            predicted_label = labels[0]  # Benign otherwise

        logger.info(f"Prediction: {predicted_label}")
        os.remove(file_path)  # Remove the uploaded file after prediction

        # Return prediction result
        return jsonify({
            'prediction': predicted_label,
            'malignant_probability': malignant_probability,
            'benign_probability': benign_probability
        })

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({'error': 'Failed to process image', 'details': str(e)}), 500


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
