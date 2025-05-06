# ========== ENVIRONMENT SETUP AND IMPORTS ==========
import os
# Suppress TensorFlow warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warnings, 3=errors

import tensorflow as tf
# Core machine learning and image processing libraries
import cv2  # OpenCV for image processing
import numpy as np  # Numerical operations
from statistics import mode  # For calculating mode of recent predictions

from tensorflow.keras.layers import Lambda
from tensorflow.keras.saving import register_keras_serializable
from keras.models import load_model  # Load saved Keras models
from keras.applications.resnet import preprocess_input  # ResNet preprocessing

# Custom utility functions from project files
from utils.datasets import get_labels  # Get emotion label names
from utils.inference import (  # Face detection and visualization functions
    detect_faces, draw_text,
    draw_bounding_box, apply_offsets,
    load_detection_model
)

# ========== CONFIGURATION SETTINGS ==========
# Path to Haar Cascade face detection model
detection_model_path = r'C:\Users\User\Documents\OsuSpring2025\DeepLearning\FProject\.venv\detection_model\haarcascade_frontalface_default.xml'
# Path to trained emotion recognition model
emotion_model_path = r'C:\Users\User\Documents\OsuSpring2025\DeepLearning\FProject\.venv\emotion_model\best_model.keras'
# Get human-readable emotion labels from FER2013 dataset
emotion_labels = get_labels('fer2013')

# Number of frames to consider for mode calculation
frame_window = 10
# Offset values for expanding face detection region (pixels)
emotion_offsets = (20, 40)

@register_keras_serializable(package="Custom", name="ResNetPreprocessLayer")
class ResNetPreprocessLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return preprocess_input(tf.cast(inputs, tf.float32))

    def get_config(self):
        config = super().get_config()
        return config

# ========== PREPROCESSING FUNCTION ==========
@register_keras_serializable(package="Custom", name="resnet_preprocess")
def resnet_preprocess(x):
    """Preprocess input for ResNet model using TensorFlow operations
    - Converts to float32
    - Applies channel-wise mean subtraction
    - Returns values in [-1, 1] range
    """
    return tf.keras.applications.resnet.preprocess_input(
        tf.cast(x, tf.float32)
    )

@register_keras_serializable(name="ReduceMeanKeepDims")
def reduce_mean_keepdims(x):
    return tf.reduce_mean(x, axis=-1, keepdims=True)

@register_keras_serializable(name="ReduceMaxKeepDims")
def reduce_max_keepdims(x):
    return tf.reduce_max(x, axis=-1, keepdims=True)

@register_keras_serializable(name="SpatialOutputShape")
def spatial_output_shape(input_shape):
    return input_shape[:-1] + (1,)


# ========== MODEL LOADING ==========
# Global variables for loaded models
face_detection = None # Haar Cascade face detector
emotion_classifier = None # Emotion recognition model

try:
    # Load face detection model (Haar Cascade classifier)
    face_detection = load_detection_model(detection_model_path)

    emotion_classifier = load_model(
        emotion_model_path,
        compile=False,
        custom_objects={
            'ResNetPreprocessLayer': ResNetPreprocessLayer,
            'ReduceMeanKeepDims': reduce_mean_keepdims,
            'ReduceMaxKeepDims': reduce_max_keepdims,
            'SpatialOutputShape': spatial_output_shape
        },
        safe_mode=False  # <-- allow Lambda deserialization
    )
except Exception as e:
    print(f"Error loading models: {e}")
    exit()

# Verify both models loaded successfully
if face_detection is None or emotion_classifier is None:
    print("Critical Error: Failed to load one or more models")
    exit()

# ========== MAIN PROCESSING FUNCTION ==========
def main():
    global face_detection, emotion_classifier

    # Initialize video capture from default camera
    video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW) # DirectShow backend for Windows
    if not video_capture.isOpened():
        print("Error: Could not open camera!")
        return

    # Get model input dimensions from loaded model
    emotion_target_size = emotion_classifier.input_shape[1:3]  # Expected (height, width)
    # Buffer for storing recent emotion predictions
    emotion_window = []

    # Full emotion color mapping in BGR format
    color_map = {
        'angry': (0, 0, 255),  # Red
        'disgust': (0, 255, 0),  # Green
        'fear': (128, 0, 128),  # Purple
        'sad': (255, 0, 0),  # Blue
        'happy': (0, 255, 255),  # Yellow
        'surprise': (255, 0, 255),  # Pink/Magenta
        'neutral': (230, 216, 173)  # Light Blue
    }

    # Main processing loop
    while True:
        # Read frame from camera
        ret, frame = video_capture.read()
        if not ret:
            break # Exit if frame capture fails

        # Convert frame to RGB for processing and grayscale for face detection
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces using Haar Cascade classifier
        faces = detect_faces(face_detection, gray_image) if face_detection else []

        # Process each detected face
        for face_coordinates in faces:
            try:
                # Apply offsets to expand face region
                x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)

                # ========== SAFETY CHECKS ==========
                # Get image dimensions
                h, w = gray_image.shape[:2]

                # Clamp coordinates to valid range
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                # Skip invalid regions
                if x1 >= x2 or y1 >= y2:
                    continue

                # Extract face region from RGB image
                face_region = frame[y1:y2, x1:x2]

                # Skip empty regions (safety check)
                if face_region.size == 0:
                    continue

                # Resize to model input size
                face_region_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
                face_resized = cv2.resize(face_region, emotion_target_size)

                # Ensure 3-channel input (convert grayscale to RGB)
                if len(face_resized.shape) == 2:
                    face_resized = np.stack([face_resized] * 3, axis=-1)

                # ========== PREPROCESSING ==========
                # Apply ResNet-specific preprocessing
                face_processed = resnet_preprocess(face_resized)
                # Add batch dimension (model expects batches)
                face_processed = np.expand_dims(face_processed, axis=0)

                # ========== EMOTION PREDICTION ==========
                predictions = emotion_classifier.predict(face_processed)
                emotion_idx = np.argmax(predictions) # Get most probable emotion
                emotion_text = emotion_labels[emotion_idx]
                emotion_prob = np.max(predictions) # Get confidence score

                # ========== UPDATE PREDICTION HISTORY ==========
                emotion_window.append(emotion_text)
                # Maintain fixed-size window of predictions
                if len(emotion_window) > frame_window:
                    emotion_window.pop(0)

                # ========== VISUALIZATION ==========
                # Get color from BGR color map and adjust intensity
                base_color = color_map.get(emotion_text, (230, 216, 173))  # Default light blue
                color = np.array(base_color) * emotion_prob
                color = color.astype(int).tolist()

                # Draw bounding box and text on RGB image
                draw_bounding_box(face_coordinates, frame, color)
                current_emotion = mode(emotion_window) if emotion_window else emotion_text
                draw_text(
                    face_coordinates, frame,
                    current_emotion,
                    color, 0, -45, 1, 2
                ) # Text position and styling

            except Exception as e:
                print(f"Face processing error: {e}")
                continue

        # Convert back to BGR for OpenCV display
        cv2.imshow('Emotion Analysis', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup resources
    video_capture.release()
    cv2.destroyAllWindows()

# ========== ENTRY POINT ==========
if __name__ == "__main__":
    main()