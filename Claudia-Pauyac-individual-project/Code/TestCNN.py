import tensorflow as tf
from tensorflow.keras import layers, Input, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns
import numpy as np

# Paths to the image dataset
# Dataset structure:
# ├── train/
# │   ├── angry/
# │   ├── disgust/
# │   ├── ...
# ├── test/
# │   ├── angry/
# │   ├── disgust/
# │   ├── ...
train_dir = r'C:\Users\User\Documents\OsuSpring2025\DeepLearning\FProject\.venv\train'
test_dir = r'C:\Users\User\Documents\OsuSpring2025\DeepLearning\FProject\.venv\test'

# Define image parameters
IMG_SIZE = 48  # FER2013 uses 48x48 images
CHANNELS = 1
BATCH_SIZE = 32
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']  # 7 classes

EPOCHS = 20

# Create data generators with augmentation for training
def prepare_data():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15, # Randomly rotates images by ±15 degrees
        width_shift_range=0.15, # Randomly shifts images horizontally by 10% of the width
        height_shift_range=0.15, # Randomly shifts images vertically by 10% of the height
        horizontal_flip=True, # Flips images horizontally
        zoom_range=0.2, # Randomly zooms in/out by up to 10%
        fill_mode='constant',  # For better handling of empty areas
        validation_split=0.2  # Split train/val
    )

    test_datagen = ImageDataGenerator(rescale = 1./255)

    # Load images from directories
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE,IMG_SIZE),
        batch_size=BATCH_SIZE,
        color_mode='grayscale',  # Use 'rgb' for color images
        class_mode='categorical',
        classes=CLASS_NAMES,
        subset='training'  # Omit if no validation split
    )

    val_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE,IMG_SIZE),
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode='categorical',
        classes=CLASS_NAMES,
        subset='validation'  # Omit if no validation split
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_SIZE,IMG_SIZE),
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode='categorical',
        classes=CLASS_NAMES,
        shuffle=False,  # Important for evaluation
    )

    # Verify class indices
    print("\nTraining Class Indices:", train_generator.class_indices)
    print("Test Class Indices:", test_generator.class_indices)

    return train_generator, val_generator, test_generator

# Build the model (Construct the CNN architecture)
def build_model():
    model = Sequential([
        # Convolutional blocks
        # 1
        layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), kernel_regularizer=regularizers.L2(0.001), padding='same',input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS)),
        layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
        layers.Activation('relu'),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.2),

        # 2
        layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), kernel_regularizer=regularizers.L2(0.001), padding='same'),
        layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
        layers.Activation('relu'),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.2),

        # 3
        layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), kernel_regularizer=regularizers.L2(0.001), padding='same'),
        layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
        layers.Activation('relu'),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.3),

        # 4
        layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), kernel_regularizer=regularizers.L2(0.001), padding='same'),
        layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
        layers.Activation('relu'),
        layers.Dropout(0.5),

        # 5
        #layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), kernel_regularizer=regularizers.L2(0.001), padding='same'),
        #layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
        #layers.Activation('relu'),
        #layers.Dropout(0.5),

        # 6
        #layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), kernel_regularizer=regularizers.L2(0.001), padding='same'),
        #layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
        #layers.Activation('relu'),
        #layers.Dropout(0.5),

        # 7
        #layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), kernel_regularizer=regularizers.L2(0.001), padding='same'),
        #layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
        #layers.Activation('relu'),
        #layers.Dropout(0.5),

        # 8
        #layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), kernel_regularizer=regularizers.L2(0.001), padding='same'),
        #layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
        #layers.Activation('relu'),
        #layers.Dropout(0.5),

        # 9
        #layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), kernel_regularizer=regularizers.L2(0.001), padding='same'),
        #layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
        #layers.Activation('relu'),
        #layers.Dropout(0.5),

        # Classifier head
        layers.GlobalAveragePooling2D(),

        # FC Layer 1
        layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.5),

        # FC Layer 2
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.5),

        # FC Layer 3
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.5),

        # Output Layer
        layers.Dense(7, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Train the model
def train_model(model, train_gen, val_gen):
    early_stop = tf.keras.callbacks.EarlyStopping(
        patience = 5,
        restore_best_weights = True
    )

    history = model.fit(
        train_gen,
        epochs = EPOCHS,
        validation_data = val_gen,
        callbacks = [early_stop]
    )

    return history

# Evaluate model performance on test set
def evaluate_model(model, test_gen):
    test_loss, test_acc = model.evaluate(test_gen,verbose=0)
    print(f'\nTest accuracy: {test_acc:.2f}')
    return test_acc


def visualize_kernels(model, layer_index=0):
    """Visualize convolutional kernels from specified layer."""
    conv_layers = [layer for layer in model.layers
                   if 'Conv2D' in str(layer.__class__)]

    if not conv_layers:
        raise ValueError("No convolutional layers found in the model")

    layer = conv_layers[layer_index]
    kernels, _ = layer.get_weights()

    # Normalize kernels for visualization
    kernels = (kernels - kernels.min()) / (kernels.max() - kernels.min())

    # Plot configuration
    n_filters = kernels.shape[-1]
    n_cols = 8
    n_rows = int(np.ceil(n_filters / n_cols))

    plt.figure(figsize=(n_cols * 2, n_rows * 2))
    for i in range(n_filters):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(kernels[:, :, 0, i], cmap='viridis')
        plt.axis('off')
    plt.suptitle(f'Convolutional Kernels from Layer {layer_index}', y=0.95)
    plt.show()

def plot_history(history):
    history_dict = history.history
    train_acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    epochs = history.epoch
    plt.plot(epochs, train_acc, 'b', epochs, val_acc, 'r')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.show()

    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss Metrics')
    plt.show()

def calculate_metrics(model, test_gen, class_names):
    """Calculate and display classification metrics and confusion matrix."""
    # Get all test data from generator
    test_gen.reset()
    y_true = []
    y_pred = []

    for _ in range(len(test_gen)):
        x, y = next(test_gen)
        y_true.extend(np.argmax(y, axis=1))
        y_pred.extend(np.argmax(model.predict(x, verbose=0), axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Verify class coverage
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    print("\nDetected Classes:", unique_classes)

    # Calculate metrics
    f1 = f1_score(y_true, y_pred, average='macro')

    # Create normalized confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize by row

    # Calculate per-class accuracy
    class_accuracies = {}
    for i, class_name in enumerate(class_names):
        class_mask = (y_true == i)
        class_total = np.sum(class_mask)
        if class_total == 0:
            print(f"Warning: No samples found for class {class_name}!")
            class_acc = 0.0
        else:
            class_correct = np.sum(y_pred[class_mask] == i)
            class_acc = class_correct / class_total

        class_accuracies[class_name] = class_acc

    # Print results with alignment
    print("\n{:<12} {:<10} {:<10}".format('Class', 'Accuracy', 'Samples'))
    for i, name in enumerate(class_names):
        samples = np.sum(y_true == i)
        acc = class_accuracies[name]
        print(f"{name.upper():<12} {acc:.2%}     {samples}")

    # Plot confusion matrix with 0-1 scale
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, vmin=0, vmax=1)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    return f1, cm_normalized, class_accuracies

# Plot the output feature maps from a specified convolutional layer.
def plot_feature_maps(model, input_image, layer_index=0, rows=4, cols=8):
    """Args:
        model (keras.Model): Trained Keras model
        input_image (np.array): Preprocessed input image (1 sample)
        layer_index (int): Index of convolutional layer to visualize
        rows (int): Number of rows in visualization grid
        cols (int): Number of columns in visualization grid"""
    # Get convolutional layers
    conv_layers = [layer for layer in model.layers
                   if isinstance(layer, layers.Conv2D)]

    if not conv_layers:
        raise ValueError("No convolutional layers found in the model")

    if layer_index >= len(conv_layers):
        raise ValueError(f"Layer index {layer_index} exceeds number of conv layers ({len(conv_layers)})")

    # Create submodel that outputs feature maps from target layer
    activation_model = tf.keras.Model(
        inputs=model.inputs,  # Use model.inputs instead of model.input
        outputs=conv_layers[layer_index].output
    )

    # Add batch dimension if needed
    if len(input_image.shape) == 3:
        input_image = np.expand_dims(input_image, axis=0)

    # Get activations
    activations = activation_model.predict(input_image, verbose=0)

    # Plot configuration
    n_filters = activations.shape[-1]
    plt.figure(figsize=(cols*2, rows*2))

    # Plot first 'rows*cols' feature maps
    for i in range(min(rows*cols, n_filters)):
        plt.subplot(rows, cols, i+1)
        plt.imshow(activations[0, :, :, i], cmap='viridis')
        plt.axis('off')
    plt.suptitle(f'Feature Maps from Layer {layer_index} ({conv_layers[layer_index].name})', y=0.92)
    plt.show()

def main():
    # Data preparation
    train_gen, val_gen, test_gen = prepare_data()

    # Model construction
    model = build_model()

    # Build model graph with dummy data
    dummy_data = np.zeros((1, IMG_SIZE, IMG_SIZE, CHANNELS))
    model.predict(dummy_data, verbose=0)  # Force input tensor initialization

    # Model training
    history = train_model(model, train_gen, val_gen)

    # Model evaluation
    test_acc = evaluate_model(model, test_gen)

    # Visualization
    plot_history(history)

    # Calculate metrics
    f1, cm, class_accuracies = calculate_metrics(model, test_gen, CLASS_NAMES)
    print(f'\nF1 macro score: {f1:.5f}')

    visualize_kernels(model, layer_index=0)
    visualize_kernels(model, layer_index=1)
    visualize_kernels(model, layer_index=2)
    visualize_kernels(model, layer_index=3)

    # Get sample image for feature map visualization
    test_gen.reset()
    sample_images, _ = next(test_gen)
    sample_image = sample_images[0]  # Get first image in batch

    # Plot feature maps from different convolutional layers
    plot_feature_maps(model, sample_image, layer_index=0)  # First conv layer
    plot_feature_maps(model, sample_image, layer_index=1)  # Second conv layer
    plot_feature_maps(model, sample_image, layer_index=2)  # Third conv layer
    plot_feature_maps(model, sample_image, layer_index=3)  # Third conv layer

if __name__ == "__main__":
    main()
