# Gesture-Classification-System

Estimate hand pose using MediaPipe (Python version).<br> This is a sample program that recognizes hand signs and finger gestures with a simple MLP using the detected key points.

<br> 
<img src="https://user-images.githubusercontent.com/37477845/102222442-c452cd00-3f26-11eb-93ec-c387c98231be.gif" width="100%" alt="Gesture Demo">

This repository contains the following contents.
* Sample program (`app.py`)
* Hand sign recognition model (TFLite)
* Finger gesture recognition model (TFLite)
* Learning data for hand sign recognition and notebook for learning
* Learning data for finger gesture recognition and notebook for learning

---

## 1. System Architecture

The application decouples hand landmark tracking from gesture classification, running a 5-layer pipeline in real-time on standard CPU edge hardware:

1.  **Hardware Layer:** Webcam captures frames at 30 FPS.
2.  **Framework Layer:** OpenCV decodes frames and Google MediaPipe Hand solution extracts 21 raw 3D hand skeleton keypoints.
3.  **Preprocessing Layer:** Translates and scales the coordinates to make them scale and translation invariant.
4.  **Inference Layer:** Separate TFLite engines execute lightweight MLP classifiers locally (sub-millisecond latency).
5.  **Presentation Layer:** OpenCV overlays bounding boxes, predicted labels, and active frame rates (FPS).

---

## 2. Requirements
* mediapipe 0.8.1
* OpenCV 3.4.2 or Later
* Tensorflow 2.3.0 or Later<br>tf-nightly 2.5.0.dev or later (Only when creating a TFLite for an LSTM model)
* scikit-learn 0.23.2 or Later (Only if you want to display the confusion matrix) 
* matplotlib 3.3.2 or Later (Only if you want to display the confusion matrix)

---

## 3. Demo
Here's how to run the demo using your webcam.
```bash
python app.py
```

The following options can be specified when running the demo.
* `--device`<br>Specifying the camera device number (Default：0)
* `--width`<br>Width at the time of camera capture (Default：960)
* `--height`<br>Height at the time of camera capture (Default：540)
* `--use_static_image_mode`<br>Whether to use static_image_mode option for MediaPipe inference (Default：Unspecified)
* `--min_detection_confidence`<br>Detection confidence threshold (Default：0.5)
* `--min_tracking_confidence`<br>Tracking confidence threshold (Default：0.5)

---

## 4. Directory Structure
<pre>
│  app.py
│  keypoint_classification.ipynb
│  point_history_classification.ipynb
│  
├─model
│  ├─keypoint_classifier
│  │  │  keypoint.csv
│  │  │  keypoint_classifier.hdf5
│  │  │  keypoint_classifier.py
│  │  │  keypoint_classifier.tflite
│  │  └─ keypoint_classifier_label.csv
│  │          
│  └─point_history_classifier
│      │  point_history.csv
│      │  point_history_classifier.hdf5
│      │  point_history_classifier.py
│      │  point_history_classifier.tflite
│      └─ point_history_classifier_label.csv
│          
└─utils
    └─cvfpscalc.py
</pre>

---

## 5. Model Training & Datasets

### A. Datasets
*   **Hand Sign (Static Keypoints):** `keypoint.csv` contains **4,787 samples** across 4 classes: `Open` (Class 0), `Close` (Class 1), `Pointer` (Class 2), and `OK` (Class 3).
*   **Finger Gesture (Point History):** `point_history.csv` contains **5,296 samples** of fingertip coordinates over 16 frames across 4 classes: `Stop` (Class 0), `Clockwise` (Class 1), `Counter Clockwise` (Class 2), and `Move` (Class 3).

### B. Evaluation Results
Both models were trained using a 70:30 train-test split:
*   **Keypoint MLP:** Validation accuracy of **~96.23%**. Output quantized TFLite model size is **~6.2 KB**.
*   **Point History MLP:** Validation accuracy of **~97.13%**. Output quantized TFLite model size is **~6.5 KB**.

---

## 6. Training Guide

### Hand sign recognition training
#### 1. Learning data collection
Press "k" to enter the mode to save key points（displayed as 「MODE:Logging Key Point」）<br>
<img src="https://user-images.githubusercontent.com/37477845/102235423-aa6cb680-3f35-11eb-8ebd-5d823e211447.jpg" width="60%"><br><br>
If you press "0" to "9", the key points will be added to "model/keypoint_classifier/keypoint.csv" as shown below.<br>
1st column: Pressed number (used as class ID), 2nd and subsequent columns: Key point coordinates<br>
<img src="https://user-images.githubusercontent.com/37477845/102345725-28d26280-3fe1-11eb-9eeb-8c938e3f625b.png" width="80%"><br><br>
The key point coordinates are the ones that have undergone translation and scale normalization.<br>
<img src="https://user-images.githubusercontent.com/37477845/102242918-ed328c80-3f3d-11eb-907c-61ba05678d54.png" width="80%">
<img src="https://user-images.githubusercontent.com/37477845/102244114-418a3c00-3f3f-11eb-8eef-f658e5aa2d0d.png" width="80%"><br><br>
In the initial state, three types of learning data are included: open hand (class ID: 0), close hand (class ID: 1), and pointing (class ID: 2). Add 3 or later, or delete existing data in the CSV to prepare custom training sets.

#### 2. Model training
Open `keypoint_classification.ipynb` in Jupyter Notebook and execute from top to bottom. To change classes, modify `NUM_CLASSES = 4` and update `model/keypoint_classifier/keypoint_classifier_label.csv`.

---

### Finger gesture recognition training
#### 1. Learning data collection
Press "h" to enter the mode to save the history of fingertip coordinates (displayed as "MODE:Logging Point History").<br>
<img src="https://user-images.githubusercontent.com/37477845/102249074-4d78fc80-3f45-11eb-9c1b-3eb975798871.jpg" width="60%"><br><br>
If you press "0" to "9", the key points will be added to "model/point_history_classifier/point_history.csv" as shown below.<br>
1st column: Pressed number (used as class ID), 2nd and subsequent columns: Coordinate history<br>
<img src="https://user-images.githubusercontent.com/37477845/102345850-54ede380-3fe1-11eb-8d04-88e351445898.png" width="80%"><br><br>
In the initial state, 4 types of learning data are included: stationary (class ID: 0), clockwise (class ID: 1), counterclockwise (class ID: 2), and moving (class ID: 3).

#### 2. Model training
Open `point_history_classification.ipynb` in Jupyter Notebook and execute from top to bottom. To change classes, modify `NUM_CLASSES = 4` and update `model/point_history_classifier/point_history_classifier_label.csv`.

---

## 7. Cloud Scaling (Microsoft Azure Proposal)

For enterprise scaling, we propose a **hybrid cloud-edge architecture** using **Azure Machine Learning (Azure ML)**:
*   **Centralized Storage:** CSV training datasets are synced to Azure Blob Storage and versioned as **Azure ML Data Assets** to allow collaborative datasets.
*   **Scale-to-Zero Compute:** Training jobs run remotely on Azure ML Compute clusters, scaling up GPU instances for model execution and scaling back to zero when idle to save costs.
*   **Model Registry & Lifecycle:** Quantized `.tflite` models are registered and versioned in the **Azure ML Model Registry** (`gesture-model:1`, `gesture-model:2`), enabling edge clients (`app.py`) to dynamically pull down updated models on startup.

---

## 8. Verification and Testing

Beyond model accuracy, the codebase includes verification testing setups:
*   **Unit Testing:** Validates that the preprocessing functions yield correct translation and scale invariance (feature coordinates remain constant irrespective of camera placement).
*   **Integration Testing:** Verifies the connection between OpenCV webcam captures, MediaPipe joint streams, and the coordinate queue logging framework.
*   **Boundary Testing:** Implements a fallback mechanism appending null coordinates (`[0, 0]`) to the point history buffer if MediaPipe tracking is lost, preventing app crashes.

---

## References & Credits
*   Google MediaPipe Developers: [Hand Pose Estimation Guide](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
*   Kazuhito Takahashi: [Original Repository Author](https://github.com/KazuhitoTakahashi/hand-gesture-recognition-using-mediapipe)
*   Nikita Kiselov: Translation and improvements contribution
*   TensorFlow Lite Developers: [Post-training Quantization Guides](https://www.tensorflow.org/lite/performance/post_training_quantization)

---

## License
This project is distributed under the [Apache License v2.0](LICENSE).
