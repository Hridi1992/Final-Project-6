# ========== ENVIRONMENT SETUP AND IMPORTS ==========
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Lambda
from tensorflow.keras.saving import register_keras_serializable
from keras.models import load_model
from keras.applications.resnet import preprocess_input

from utils.datasets import get_labels
from utils.inference import (
    detect_faces, draw_text,
    draw_bounding_box, apply_offsets,
    load_detection_model
)

# ========== CONFIGURATION SETTINGS ==========
detection_model_path = r'C:\Users\User\Documents\OsuSpring2025\DeepLearning\FProject\.venv\detection_model\haarcascade_frontalface_default.xml'
emotion_model_path = r'C:\Users\User\Documents\OsuSpring2025\DeepLearning\FProject\.venv\emotion_model\best_model.keras'
emotion_labels = get_labels('fer2013')
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
face_detection = None
emotion_classifier = None

try:
    face_detection = load_detection_model(detection_model_path)
    emotion_classifier = load_model(
        emotion_model_path,
        compile=False,
        custom_objects={
            'ResNetPreprocessLayer': ResNetPreprocessLayer,
            'ReduceMeanKeepDims': reduce_mean_keepdims,
            'ReduceMaxKeepDims': reduce_max_keepdims,
            'SpatialOutputShape': spatial_output_shape
        }
    )
except Exception as e:
    print(f"Error loading models: {e}")
    exit()

if face_detection is None or emotion_classifier is None:
    print("Critical Error: Failed to load models")
    exit()

# ========== IMAGE PROCESSING FUNCTION ==========
def process_image(image_path, output_suffix='_processed'):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    output_image = image.copy()
    faces = detect_faces(face_detection, gray_image)

    color_map = {
        'angry': (0, 0, 255),
        'disgust': (0, 255, 0),
        'fear': (128, 0, 128),
        'sad': (255, 0, 0),
        'happy': (0, 255, 255),
        'surprise': (255, 0, 255),
        'neutral': (230, 216, 173)
    }

    for face_coords in faces:
        try:
            x1, x2, y1, y2 = apply_offsets(face_coords, emotion_offsets)
            h, w = gray_image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x1 >= x2 or y1 >= y2:
                continue

            face_region = image[y1:y2, x1:x2]
            face_resized = cv2.resize(face_region, emotion_classifier.input_shape[1:3])

            if len(face_resized.shape) == 2:
                face_resized = np.stack([face_resized] * 3, axis=-1)

            processed_face = resnet_preprocess(face_resized)
            prediction = emotion_classifier.predict(np.expand_dims(processed_face, 0))
            emotion_idx = np.argmax(prediction)
            emotion_text = emotion_labels[emotion_idx]
            emotion_prob = np.max(prediction)

            color = np.array(color_map[emotion_text]) * emotion_prob
            color = color.astype(int).tolist()

            draw_bounding_box(face_coords, output_image, color)
            draw_text(
                face_coords, output_image,
                f"{emotion_text} ({emotion_prob:.0%})",
                color, y_offset=-10, font_scale=0.7, thickness=1
            )

        except Exception as e:
            print(f"Error processing face: {str(e)}")
            continue

    output_path = os.path.splitext(image_path)[0] + output_suffix + '.jpg'
    cv2.imwrite(output_path, output_image)
    print(f"Processed image saved to: {output_path}")
    return output_path

# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    input_image = "examples/a.jpg"  # Replace with your image path
    result_path = process_image(input_image)

    if result_path:
        result = cv2.imread(result_path)
        cv2.imshow("Emotion Analysis Result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
