# fetch_model.py
"""
Azure ML Model Registry Synchronizer
Retrieves the latest version of the gesture model and copies it locally.
Fallback to cached local model if connection fails.
"""
import os
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

def sync_latest_model():
    print("Initializing Azure ML Registry connection...")
    try:
        ml_client = MLClient(
            credential=DefaultAzureCredential(),
            subscription_id="your-subscription-id",
            resource_group_name="your-resource-group",
            workspace_name="your-workspace-name"
        )
        
        # Retrieve latest model metadata
        model_name = "gesture-keypoint-model"
        latest_model = ml_client.models.get(name=model_name, label="latest")
        print(f"Latest registered model found: {model_name} (Version: {latest_model.version})")
        
        # Target local directories
        local_dir = os.path.join("..", "model", "keypoint_classifier")
        os.makedirs(local_dir, exist_ok=True)
        
        # Download model artifact
        print(f"Downloading model '{model_name}' from Azure ML Model Registry...")
        ml_client.models.download(
            name=model_name, 
            version=latest_model.version, 
            download_path=local_dir
        )
        
        # Rename downloaded file to overwrite local cached model
        downloaded_file = os.path.join(local_dir, model_name, "keypoint_classifier.tflite")
        target_file = os.path.join(local_dir, "keypoint_classifier.tflite")
        
        if os.path.exists(downloaded_file):
            os.replace(downloaded_file, target_file)
            print(f"Model successfully updated locally at: {target_file}")
            
            # Clean up the extra downloaded folder structure
            extra_folder = os.path.join(local_dir, model_name)
            if os.path.exists(extra_folder):
                os.rmdir(extra_folder)
        else:
            print("Failed to locate downloaded model binary. Using cached local model.")
            
    except Exception as e:
        print(f"Azure Connection failed: {e}")
        print("Falling back to local cached model.")

if __name__ == "__main__":
    sync_latest_model()
