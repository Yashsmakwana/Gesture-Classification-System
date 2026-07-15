# train_keypoint.py
"""
Cloud-compatible Training Script for Hand Sign (Keypoint) Classifier
Runs inside Azure ML Compute instance/job environment.
Logs metrics automatically using MLflow.
"""
import os
import argparse
import numpy as np
import tensorflow as tf
import mlflow

def main():
    # 1. Parse arguments (Azure passes input paths dynamically)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to input CSV training data")
    args = parser.parse_args()

    # 2. Initialize MLflow Auto-logging for Tensorflow
    mlflow.tensorflow.autolog()
    print("MLflow tracking initialized successfully.")

    # 3. Create outputs directory
    # Any files written to the standard './outputs' directory are collected as job artifacts by Azure ML
    os.makedirs("outputs", exist_ok=True)

    # 4. Load dataset
    print(f"Loading training data from: {args.data_path}")
    dataset = np.loadtxt(args.data_path, delimiter=',')
    X = dataset[:, 1:]
    y = dataset[:, 0].astype(int)

    # Calculate number of classes from labels
    num_classes = len(np.unique(y))
    print(f"Dataset Loaded. Samples: {X.shape[0]}, Features: {X.shape[1]}, Classes: {num_classes}")

    # 5. Define Model Architecture (Lightweight Multi-layer Perceptron)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input((X.shape[1],)),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # 6. Train Model
    # MLflow automatically logs loss, accuracy, and val metrics per epoch to Azure ML Dashboard
    model.fit(
        X, y,
        epochs=100,
        batch_size=128,
        validation_split=0.2,
        verbose=1
    )

    # 7. Convert and Quantize model to TFLite format
    print("Converting model to quantized TensorFlow Lite format...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Enable post-training float16 quantization for CPU performance
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # Save output to outputs folder
    tflite_path = "outputs/keypoint_classifier.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"Quantized TFLite model successfully saved to: {tflite_path}")

if __name__ == "__main__":
    main()
