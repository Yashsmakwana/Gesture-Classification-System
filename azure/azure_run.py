# azure_run.py
"""
Azure ML Command Job Runner
Use this script to submit a model training job to Azure Machine Learning.
Prerequisites:
    pip install azure-ai-ml azure-identity
"""
import sys
from azure.ai.ml import MLClient, command, Input
from azure.ai.ml.entities import AmlCompute, Model
from azure.identity import DefaultAzureCredential

def main():
    # 1. Connect to Azure ML Workspace
    try:
        ml_client = MLClient(
            credential=DefaultAzureCredential(),
            subscription_id="your-subscription-id",
            resource_group_name="your-resource-group",
            workspace_name="your-workspace-name"
        )
        print("Connected to Azure ML Workspace successfully.")
    except Exception as e:
        print(f"Failed to connect to Azure Workspace: {e}")
        sys.exit(1)

    # 2. Define/Retrieve Compute Target
    compute_name = "cpu-cluster"
    try:
        cpu_compute = ml_client.compute.get(compute_name)
        print(f"Using existing compute target: {compute_name}")
    except Exception:
        print(f"Creating a new compute target: {compute_name}...")
        cpu_compute = AmlCompute(
            name=compute_name,
            type="amlcompute",
            size="STANDARD_DS11_V2",
            min_instances=0,  # Scale down to zero when idle to save costs
            max_instances=2,
            idle_time_before_scale_down=120
        )
        ml_client.compute.begin_create_or_update(cpu_compute).result()
        print("Compute target created successfully.")

    # 3. Configure and Submit the Training Job
    job = command(
        code="../",  # Uploads parent folder containing classifiers and utils
        command="python azure/train_keypoint.py --data_path ${{inputs.dataset}}",
        inputs={
            "dataset": Input(
                type="uri_file",
                path="../model/keypoint_classifier/keypoint.csv"  # Sync local dataset to cloud
            )
        },
        environment="AzureML-tensorflow-2.12-ubuntu20.04-py38-cpu:1",  # Azure curated runtime environment
        compute=compute_name,
        display_name="gesture-classification-training-run",
        experiment_name="gesture-keypoints"
    )

    print("Submitting training job to Azure ML...")
    returned_job = ml_client.create_or_update(job)
    print(f"Job submitted successfully!")
    print(f"Monitor training in Azure ML Studio: {returned_job.studio_url}")

    # 4. Model Registry Registration
    # Once the job completes, the registered model will point to the TFLite binary outputted in artifacts
    model_path = f"azureml://jobs/{returned_job.name}/outputs/artifacts/paths/outputs/keypoint_classifier.tflite"
    gesture_model = Model(
        path=model_path,
        name="gesture-keypoint-model",
        description="TFLite model trained on MediaPipe keypoints for static gesture classification.",
        type="custom_model"
    )
    ml_client.models.create_or_update(gesture_model)
    print("Model registration initiated in Azure ML Model Registry.")

if __name__ == "__main__":
    main()
