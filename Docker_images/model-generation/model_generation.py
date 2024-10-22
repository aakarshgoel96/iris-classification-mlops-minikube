import joblib
import json
import boto3
from botocore.client import Config
import os

import random
from typing import Dict, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import datetime
import uuid
import logging
import mlflow
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure MLflow
mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI', 'http://mlflow-service:5000'))
mlflow.set_experiment("iris_classification")


def register_model_if_best(run_id, model_version, accuracy, best_accuracy):
    """
    Register the model in the MLflow Model Registry if it has the highest accuracy.
    
    Parameters:
        run_id (str): The run ID of the current run.
        model_version (str): The version of the current model.
        accuracy (float): The accuracy of the current model.
        best_accuracy (float): The best accuracy recorded so far.
    
    Returns:
        float: The updated best accuracy.
    """
    client = mlflow.tracking.MlflowClient()
    model_name = "BestIrisModel"
    
    # If the current accuracy is the best, register the model
    if accuracy > best_accuracy:
        print(f"New best model found with accuracy: {accuracy:.4f}, registering model...")

        # Register the model with a specific name in the MLflow Model Registry
        model_uri = f"runs:/{run_id}/model"
        registered_model = mlflow.register_model(model_uri, model_name)
        
        # Transition the registered model to the "Staging" or "Production" stage (optional)
        client.transition_model_version_stage(
            name=model_name,
            version=registered_model.version,
            stage="Production"  # You can use "Staging" or "Production"
        )
        best_accuracy = accuracy  # Update the best accuracy
    else:
        print(f"Current model accuracy {accuracy:.4f} is not better than the best accuracy {best_accuracy:.4f}.")
    
    return best_accuracy


def plot_and_log_accuracy_over_time(experiment_name="iris_classification"):
    """
    Retrieve accuracy from each run in the given MLflow experiment, plot accuracy over time,
    and log the plot as an artifact in MLflow.
    
    Parameters:
        experiment_name (str): The name of the MLflow experiment.
    """
    
    # Get the experiment details (ID) from the experiment name
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Experiment '{experiment_name}' not found.")
        return
    
    experiment_id = experiment.experiment_id
    
    # Fetch all runs in the experiment
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(experiment_ids=[experiment_id], order_by=["start_time ASC"])
    
    # Store timestamps and accuracy values
    accuracy_records = []
    
    for run in runs:
        run_id = run.info.run_id
        
        # Get the accuracy metric from each run
        accuracy = run.data.metrics.get("accuracy")
        if accuracy is not None:
            # Get the start time of the run (which can be used as the time for the plot)
            run_start_time = run.info.start_time
            run_start_time = pd.to_datetime(run_start_time, unit='ms')  # Convert from ms to datetime
            
            accuracy_records.append((run_start_time, accuracy))
    
    if not accuracy_records:
        print("No accuracy data found in the experiment runs.")
        return
    
    # Sort the accuracy records by time
    accuracy_records.sort(key=lambda x: x[0])
    
    # Convert to DataFrame for easy plotting
    df_accuracy = pd.DataFrame(accuracy_records, columns=["timestamp", "accuracy"])
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(df_accuracy["timestamp"], df_accuracy["accuracy"], marker='o', color='b', linestyle='-', markersize=5)
    plt.xlabel('Time')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy Over Time for Experiment: {experiment_name}')
    plt.grid(True)
    
    # Save the plot locally
    plot_file = "accuracy_over_time.png"
    plt.savefig(plot_file)
    plt.close()  # Close the plot after saving to avoid displaying it

    # Log the plot to MLflow as an artifact
    mlflow.log_artifact(plot_file)
    
    # Remove the local plot file if you want to clean up
    if os.path.exists(plot_file):
        os.remove(plot_file)

def train_random_forest(df: pd.DataFrame,
                        target_column: str,
                        hyperparameters: Dict[str, int]):
    """
    Train a Random Forest Classifier on the given dataset.

    Parameters:
        df (pd.DataFrame): The input dataframe containing features and target.
        target_column (str): The name of the target column in the dataframe.
        hyperparameters (dict): A dictionary of hyperparameters for the RandomForestClassifier.

    Returns:
        model: The trained Random Forest model.
        accuracy: The accuracy score of the model on the test set.
    """

    # Separate features and target from the DataFrame
    X = df.drop(target_column, axis=1)  # Features
    y = df[target_column]  # Target

    # Split the dataset into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

    # Standardize the features (optional but recommended for some models, not strictly necessary for Random Forest)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create and train the Random Forest Classifier model with hyperparameters
    model = RandomForestClassifier(**hyperparameters)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)

    # Print or log the actual accuracy to check its precision
    print(f"Accuracy: {accuracy:.4f}")  # Log accuracy to 4 decimal places for clarity

    return model, accuracy


def generate_unique_version():
    """Generate a unique version string based on timestamp and UUID."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]  # Use first 8 characters of UUID
    return f"model_v{timestamp}_{unique_id}"
 
# Function to save a model 
def save_model(model, version, hyperparameters, accuracy):
    # Save model locally
    model_file = f"{version}.pkl"
    joblib.dump(model, model_file)

    # Create metadata
    metadata = {
        "version": version,
        "hyperparameters": hyperparameters,
        "accuracy": accuracy
    }
    metadata_file = f"{version}_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f)

    # Upload to MinIO
    upload_to_minio(model_file, f"models/{model_file}")
    upload_to_minio(metadata_file, f"models/{metadata_file}")

def upload_to_minio(local_file, s3_file):
    # Configure MinIO client
    s3_client = boto3.client(
    's3',
    endpoint_url='http://minio-service.virtual-mind-task.svc.cluster.local:9000',  # Use http:// or https:// prefix
    aws_access_key_id=os.environ.get('MINIO_ACCESS_KEY'),
    aws_secret_access_key=os.environ.get('MINIO_SECRET_KEY'),
    config=Config(signature_version='s3v4'),
    region_name='us-east-1',  # Can be any valid region, MinIO doesn't use it
    verify=False  # Set to True if using HTTPS and have proper certificates
    )
    s3_client.upload_file(local_file, 's3-bucket', s3_file)


def main():
    logging.info("Starting model training")
    try:
        best_accuracy = 0.5
        
        with mlflow.start_run() as run:
            df = pd.read_csv('/app/iris.csv')
            hyperparameters = {
                'n_estimators': random.randint(1, 100),
                'max_depth': random.randint(1, 20),
                'min_samples_split': random.randint(2, 10),
                'min_samples_leaf': random.randint(1, 5),
                'random_state': random.randint(1, 1000),
            }
            
            # Log DataFrame to MLflow
            df_file = "iris_dataset.csv"
            df.to_csv(df_file, index=False)
            mlflow.log_artifact(df_file)
            
            model, accuracy = train_random_forest(df, 'variety', hyperparameters)
            model_version = generate_unique_version()
            
            # Log parameters and metrics
            mlflow.log_params(hyperparameters)
            mlflow.log_metric("accuracy", accuracy)
            
            # Save model locally first
            model_file = f"{model_version}.pkl"
            metadata_file = f"{model_version}_metadata.json"
            joblib.dump(model, model_file)
            with open(metadata_file, 'w') as f:
                json.dump({
                    "version": model_version,
                    "hyperparameters": hyperparameters,
                    "accuracy": accuracy,
                    "timestamp": datetime.datetime.now().isoformat()
                }, f)
            
            # Log model to MLflow
            mlflow.sklearn.log_model(model, "model")
            
            # Upload to MinIO
            upload_to_minio(model_file, f"models/{model_file}")
            upload_to_minio(metadata_file, f"models/{metadata_file}")
            
            logging.info(f"Model version: {model_version}")
            logging.info(f"The accuracy of the model is: {accuracy}")
            logging.info(f"Hyperparameters used: {hyperparameters}")
            
            # Register the model if it has the best accuracy
            best_accuracy = register_model_if_best(run.info.run_id, model_version, accuracy, best_accuracy)
            
            plot_and_log_accuracy_over_time("iris_classification")
            
    except Exception as e:
        logging.error(f"An error occurred during model training: {str(e)}")
        raise

if __name__ == '__main__':
    main()
