from flask import Flask, request, jsonify
import mlflow
import os
import sys
import numpy as np

app = Flask(__name__)

# Configure MLflow
mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI', 'http://mlflow-service:5000'))
mlflow.set_experiment("iris_classification")

def load_model():
    model_name = "BestIrisModel"
    
    try:
        # Get latest model version (no stage filter, just by name)
        client = mlflow.tracking.MlflowClient()
        model_version_info = client.get_latest_versions(model_name)  # Fetch latest versions without stage filter

        if not model_version_info:
            print(f"No versions found for model '{model_name}'.")
            return None
        
        # Get the most recent version number (latest version is the first entry in the list)
        latest_version = model_version_info[0].version
        model_uri = f"models:/{model_name}/{latest_version}"  # Build the model URI
        
        # Load the model using the model URI
        model = mlflow.sklearn.load_model(model_uri)
        print("Model loaded successfully from:", model_uri)
        return model
    except mlflow.exceptions.MlflowException as e:
        print(f"Failed to load the model: {str(e)}")
        return str(e)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return str(e)

# Load the model when the app starts
model = load_model()

# Test route to check if the service is running
@app.route('/test', methods=['GET'])
def test_service():
    return jsonify({"status": "Service is up and running!"})

@app.route('/predict', methods=['POST'])
def predict():
    global model
    # Attempt to reload the model in case it has changed
    new_model = load_model()
    if new_model is not None:
        model = new_model
    
    # Check if model is a string (indicating an error occurred during loading)
    if isinstance(model, str):
        return jsonify({"error": f"Model loading failed: {model}"}), 500

    # Get the incoming data
    data = request.get_json()
    print(f"Received data: {data}")  # Log incoming data for debugging

    # Validate input
    if 'input_vector' not in data:
        return jsonify({'error': 'Input vector is required'}), 400
    
    # Check if input_vector is in the correct format
    input_vector = data['input_vector']
    if not isinstance(input_vector, list) or not all(isinstance(x, (int, float)) for x in input_vector):
        return jsonify({'error': 'Input vector must be a list of numbers'}), 400

    # Make prediction
    try:
        prediction = model.predict([input_vector])
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        print(f"Prediction error: {str(e)}")  
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)  # Change port to 8080 to match Kubernetes deployment
