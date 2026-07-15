# Cloud Integration Proposal: Scaling Gesture Classification with Microsoft Azure

This proposal outlines the strategy for integrating **Microsoft Azure AI and Machine Learning (Azure ML)** into the Gesture Classification System. The proposed integration addresses the limitations of local training (lack of version control, manual run tracking, and resource constraints) by establishing a **hybrid cloud-edge architecture**.

Under this design, high-frequency, low-latency tasks (video capture, hand landmark extraction, and inference) remain on the local edge client to maintain interactive frame rates (30+ FPS), while training, model versioning, and dataset management are offloaded to Azure.

---

## 1. System Architecture

The hybrid cloud-edge architecture divides responsibilities between the local device (for real-time execution) and Azure ML (for data storage, training, and registry).

```mermaid
graph TD
    subgraph Local Client (Edge)
        A["Webcam Video Capture"] --> B["MediaPipe Hand Landmark Extraction"]
        B --> C["Landmark Coordinates (42 floats)"]
        C --> D["Local app.py Inference"]
        D --> E["Real-Time UI Display"]
        F["Data Logging Mode"] -->|Appends CSVs| G["Local keypoint.csv / point_history.csv"]
    end

    subgraph Azure Cloud
        G -->|Uploads via SDK/CLI| H[("Azure Blob Storage (Data Assets)")]
        H -->|Inputs to Job| I["Azure ML Training Job"]
        I -->|Runs train.py on Compute Cluster| J["Trained .tflite Model"]
        J -->|Registers versioned model| K["Azure ML Model Registry"]
        K -->|Syncs/Downloads| D
    end
```

---

## 2. Core Azure Integration Components

### A. Centralized Data Storage (Azure Blob Storage)
Currently, training data is stored in flat CSV files (`keypoint.csv` and `point_history.csv`) on the developer's local machine. 
*   **Azure Solution:** Register these CSV files as **Azure ML Data Assets** backed by **Azure Blob Storage**.
*   **Benefit:** Centralizing the datasets allows multiple developers or clients to contribute training data. Azure ML keeps a history of dataset versions, allowing you to reproduce older training runs.

### B. Scalable Cloud Training (Azure ML Command Jobs)
Currently, training is executed in local Jupyter Notebooks using local CPU/GPU resources.
*   **Azure Solution:** Convert the training notebooks into clean, parameterized Python training scripts (e.g., `train_keypoint.py`). Submit these as **Azure ML Command Jobs** using the Azure ML SDK v2.
*   **Benefit:** Training runs on dedicated, on-demand compute resources in Azure (compute clusters). You can scale the compute up to high-performance GPU virtual machines for complex models (e.g., training LSTM configurations) and configure them to **scale to zero** when idle to eliminate unnecessary cloud costs.

### C. Experiment Tracking & Metrics (Azure ML MLflow Integration)
*   **Azure Solution:** Integrate MLflow into the training script. During execution on Azure compute, training accuracy, validation loss, confusion matrices, and model graphs are logged automatically.
*   **Benefit:** Provides a centralized dashboard in Azure ML Studio to compare different training runs, compare hyperparameters, and analyze training logs.

### D. Model Lifecycle Management (Azure ML Model Registry)
*   **Azure Solution:** Register the compiled `.tflite` model directly into the **Azure ML Model Registry** upon successful training.
*   **Benefit:** The Model Registry maintains a clear history of model versions (e.g., `gesture-model:1`, `gesture-model:2`). The local client application (`app.py`) can dynamically fetch the latest approved model version via a lightweight SDK script prior to execution.

---

## 3. Code Implementation Guide

Below is the conceptual implementation using the modern **Azure ML Python SDK v2** (`azure-ai-ml`).

### Script 1: Local Automation Script (`azure_run.py`)
This script runs on the developer's local machine to orchestrate uploading the dataset, launching the training job on Azure compute, and registering the resulting model.

```python
# azure_run.py
from azure.ai.ml import MLClient, command, Input, Output
from azure.ai.ml.entities import AmlCompute, Model
from azure.identity import DefaultAzureCredential

# 1. Connect to Azure ML Workspace
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id="your-subscription-id",
    resource_group_name="your-resource-group",
    workspace_name="your-workspace-name"
)

# 2. Define Compute Target (Scale-to-zero CPU cluster)
compute_name = "cpu-cluster"
try:
    cpu_compute = ml_client.compute.get(compute_name)
    print(f"Compute target '{compute_name}' already exists.")
except Exception:
    print(f"Creating a new compute target: {compute_name}...")
    cpu_compute = AmlCompute(
        name=compute_name,
        type="amlcompute",
        size="STANDARD_DS11_V2",
        min_instances=0,  # Scale to zero when idle
        max_instances=2,
        idle_time_before_scale_down=120
    )
    ml_client.compute.begin_create_or_update(cpu_compute).result()

# 3. Configure and Submit the Training Job
job = command(
    code="./",  # Uploads current directory including model/ and train_keypoint.py
    command="python train_keypoint.py --data_path ${{inputs.dataset}}",
    inputs={
        "dataset": Input(
            type="uri_file",
            path="model/keypoint_classifier/keypoint.csv"  # Local dataset path
        )
    },
    environment="AzureML-tensorflow-2.12-ubuntu20.04-py38-cpu:1",  # Azure curated environment
    compute=compute_name,
    display_name="gesture-classification-training-run",
    experiment_name="gesture-keypoints"
)

print("Submitting training job to Azure ML...")
returned_job = ml_client.create_or_update(job)
print(f"Job submitted! Studio Web URL: {returned_job.studio_url}")

# 4. Register the output TFLite model after job completion
# (This can also be done automatically at the end of the training script)
model_path = f"azureml://jobs/{returned_job.name}/outputs/artifacts/paths/outputs/keypoint_classifier.tflite"
gesture_model = Model(
    path=model_path,
    name="gesture-keypoint-model",
    description="TFLite model trained on hand keypoints for gesture classification.",
    type="custom_model"
)
ml_client.models.create_or_update(gesture_model)
print("Model registered in Azure ML Model Registry.")
```

### Script 2: Cloud-Compatible Training Script (`train_keypoint.py`)
This script represents the converted notebook logic running inside the Azure ML compute environment.

```python
# train_keypoint.py
import argparse
import os
import numpy as np
import tensorflow as tf
import mlflow

# Start MLflow auto-logging to capture Keras training metrics in Azure Studio
mlflow.tensorflow.autolog()

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, help="Path to input CSV training data")
args = parser.parse_args()

# Create directory to save the outputs
os.makedirs("outputs", exist_ok=True)

# Load dataset
dataset = np.loadtxt(args.data_path, delimiter=',')
X = dataset[:, 1:]
y = dataset[:, 0].astype(int)

# Train-Test Split & Preprocessing (Standard scikit-learn logic)
# ... [Model Architecture Definition] ...
model = tf.keras.models.Sequential([
    tf.keras.layers.Input((X.shape[1],)),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  # Assumes 3 classes
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train Model (MLflow automatically records accuracy/loss epochs)
model.fit(X, y, epochs=100, batch_size=128, validation_split=0.2)

# Convert to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model to the Azure-designated "outputs" directory
# Any file saved in "outputs" is collected as a job artifact
tflite_path = "outputs/keypoint_classifier.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print(f"Model saved successfully to {tflite_path}")
```

### Script 3: Local Model Sync Script (`fetch_model.py`)
This script runs locally on the edge client prior to executing `app.py`. It pulls the latest version of the model from the Azure cloud registry.

```python
# fetch_model.py
import os
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

def sync_latest_model():
    print("Connecting to Azure ML Registry...")
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id="your-subscription-id",
        resource_group_name="your-resource-group",
        workspace_name="your-workspace-name"
    )
    
    # Retrieve the latest model metadata
    model_name = "gesture-keypoint-model"
    latest_model = ml_client.models.get(name=model_name, label="latest")
    print(f"Latest model version: {latest_model.version}")
    
    # Target local download path
    local_dir = "model/keypoint_classifier"
    os.makedirs(local_dir, exist_ok=True)
    
    # Download model artifact
    print(f"Downloading model '{model_name}' (version {latest_model.version}) from Azure...")
    ml_client.models.download(
        name=model_name, 
        version=latest_model.version, 
        download_path=local_dir
    )
    
    # Move/rename model to match the app's expectations
    downloaded_file = os.path.join(local_dir, model_name, "keypoint_classifier.tflite")
    target_file = os.path.join(local_dir, "keypoint_classifier.tflite")
    if os.path.exists(downloaded_file):
        os.replace(downloaded_file, target_file)
        print(f"Model successfully synchronized to {target_file}")

if __name__ == "__main__":
    try:
        sync_latest_model()
    except Exception as e:
        print(f"Failed to fetch model from Azure: {e}. Falling back to cached local model.")
```

---

## 4. Key Metrics and Results Tracking

Once integrated, Azure Machine Learning Studio provides an interactive dashboard to trace system telemetry:
1.  **Run Comparison:** Side-by-side comparison of hyperparameters (learning rates, batch sizes, dense layers) against validation accuracy.
2.  **Dataset Tracking:** Complete linage indicating which version of `keypoint.csv` was used to train which version of the model.
3.  **Cost Monitoring:** Built-in charts displaying runtimes and cost metrics for the compute resources consumed.
