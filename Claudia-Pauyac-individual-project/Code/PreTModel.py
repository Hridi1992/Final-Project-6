import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from collections import defaultdict
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.saving import register_keras_serializable
from sklearn.metrics import (
    f1_score,
    classification_report,
    cohen_kappa_score,
    log_loss,
    precision_score,
    top_k_accuracy_score,
    roc_auc_score,
    confusion_matrix,
    balanced_accuracy_score  # Added here
)

# Define paths
train_dir = r'C:\Users\User\Documents\OsuSpring2025\DeepLearning\FProject\.venv\train'  # Contains class subfolders
test_dir = r'C:\Users\User\Documents\OsuSpring2025\DeepLearning\FProject\.venv\test'

# Image parameters
IMG_SIZE = 48  # Original FER2013 size
TARGET_SIZE = 224  # ResNet50 input size
BATCH_SIZE = 32
# FER2013 class names (ensure this matches your label mapping)
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']


@register_keras_serializable()
def resnet_preprocess(x):
    return preprocess_input(x)


def print_class_distribution(data_dir):
    """Print number of images per class in a directory"""
    CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    print("\n=== Class Distribution ===")
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(data_dir, class_name)
        if os.path.exists(class_dir):
            num_images = len(os.listdir(class_dir))
            print(f"{class_name.capitalize():<9}: {num_images} images")
        else:
            print(f"{class_name.capitalize():<9}: Directory not found")


def create_data_generators():
    """Create and configure data generators for training, validation, and testing"""
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_NAMES,
        subset='training'
    )

    val_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_NAMES,
        shuffle=False,
        subset='validation'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_NAMES,
        shuffle=False
    )

    return train_generator, val_generator, test_generator


def visualize_conv_kernels(model, layer_name, layer_index, max_filters=32):
    """
    Visualizes the kernels of a Conv2D layer in the model.

    Args:
        model: Keras model
        layer_name: Name of the Conv2D layer to visualize
        max_filters: Maximum number of filters to display (None for all)
    """
    if (max_filters == -1):
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
    else:
        # Retrieve the layer by name
        try:
            layer = model.get_layer(layer_name)
        except ValueError:
            print(f"Layer '{layer_name}' not found in the model.")
            return

        # Check if it's a Conv2D layer
        if not isinstance(layer, layers.Conv2D):
            print(f"Layer '{layer_name}' is not a Conv2D layer.")
            return

        # Get the kernels (weights)
        kernels = layer.get_weights()[0]

        # Normalize to [0, 1] for visualization
        kernels = (kernels - kernels.min()) / (kernels.max() - kernels.min())

        # Get number of filters and channels
        num_filters = kernels.shape[-1]
        num_channels = kernels.shape[-2]  # Input channels

        # Limit displayed filters
        if max_filters is not None:
            num_filters = min(num_filters, max_filters)

        # Calculate grid dimensions
        cols = 8
        rows = int(np.ceil(num_filters / cols))

        # Create figure
        fig = plt.figure(figsize=(cols * 2, rows * 2))
        plt.suptitle(f"Kernels from layer: {layer_name}", fontsize=16)

        # Plot each kernel
        for i in range(num_filters):
            ax = fig.add_subplot(rows, cols, i + 1)

            # Get kernel (height, width, input_channels)
            kernel = kernels[:, :, :, i]

            # For RGB kernels (3 input channels)
            if kernel.shape[-1] == 3:
                ax.imshow(kernel)
            else:
                # For grayscale/other channels (take mean)
                ax.imshow(np.mean(kernel, axis=-1), cmap='gray')

            ax.axis('off')

        plt.tight_layout()
        plt.show()


# 2. Build model with integrated preprocessing
def build_resnet_model():
    # Input and augmentation
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))

    # Data augmentation
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.15)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.RandomContrast(0.1)(x)

    # ResNet50 preprocessing
    x = layers.Resizing(TARGET_SIZE, TARGET_SIZE)(x)
    x = layers.Concatenate()([x, x, x])  # Grayscale to RGB
    x = layers.Lambda(resnet_preprocess)(x)  # Official ResNet50 preprocessing

    # Base model
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(TARGET_SIZE, TARGET_SIZE, 3),
        name='resnet50'
    )
    x = base_model(x)

    print(f"Total layers in ResNet50: {len(base_model.layers)}")

    # Classifier
    # New: Additional Conv Layer
    x = layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), kernel_regularizer=regularizers.L2(0.001),
                      padding='same')(x)
    x = layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)

    x = layers.GlobalAveragePooling2D()(x)

    # New: Additional Dense Layer
    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.L2(0.001))(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(7, activation='softmax')(x)

    return models.Model(inputs, outputs)


# --------------------------------------------------
# 3. Initial Training Phase (Frozen Base Model) and Fine-Tuning Phase (Partial Unfreezing)
# --------------------------------------------------
def train_model(model, train_gen, val_gen, initial_epochs=1, fine_tune_epochs=1):
    """Handle both initial training and fine-tuning phases"""

    # Initial training with frozen base
    print("\n=== Initial Training ===")
    model.get_layer("resnet50").trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    initial_history = model.fit(
        train_gen,
        epochs=initial_epochs,
        validation_data=val_gen,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint('initial_best.keras', save_best_only=True)
        ]
    )

    # Fine-tuning
    print("\n=== Fine-Tuning ===")
    base_model = model.get_layer("resnet50")
    base_model.trainable = True

    # Freeze first 140 layers (80% of total)
    for layer in base_model.layers[:140]:
        layer.trainable = False

    # Unfreeze last 35 layers (conv5_x blocks + top)
    for layer in base_model.layers[140:]:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-5, decay_steps=1000, decay_rate=0.9)),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    fine_tune_history = model.fit(
        train_gen,
        epochs=initial_epochs + fine_tune_epochs,
        initial_epoch=initial_history.epoch[-1] + 1,
        validation_data=val_gen,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=3),
            tf.keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True)
        ]
    )

    return model


def evaluate_model(model, test_gen):
    """Evaluate model performance with accuracy, F1 score, and confusion matrix"""
    # Get all test data from generator
    test_gen.reset()
    y_true = []
    y_pred = []
    y_pred_probs = []
    for _ in range(len(test_gen)):
        x, y = next(test_gen)
        y_true.extend(np.argmax(y, axis=1))
        batch_pred_probs = model.predict(x, verbose=0)
        y_pred_probs.extend(batch_pred_probs)
    y_true = np.array(y_true)
    y_pred_probs = np.vstack(y_pred_probs)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\n=== Final Evaluation ===")

    # Basic evaluation
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"Test Accuracy: {test_acc:.2%}")

    # Generate predictions
    # y_pred_probs = model.predict(test_gen)
    # y_pred = np.argmax(y_pred_probs, axis=1)
    # y_true = test_gen.classes

    # F1 Score
    f1 = f1_score(y_true, y_pred, average='macro')
    print(f"F1 Macro Score: {f1:.2%}")

    # Classification report
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))

    # Balanced accuracy
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    print(f"Balanced accuracy Score: {balanced_acc:.2%}")

    # Cohen's Kappa
    kappa = cohen_kappa_score(y_true, y_pred)
    print(f"Cohen Kappa Score: {kappa:.2%}")

    # ROC AUC (Macro/Micro)
    roc_auc = roc_auc_score(y_true, y_pred_probs, multi_class='ovr', average='macro')
    print(f"ROC AUC Score: {roc_auc:.2%}")

    # Top-k accuracy
    top2_acc = top_k_accuracy_score(y_true, y_pred_probs, k=2)
    print(f"Top-k Score: {top2_acc:.2%}")

    # Class-specific metrics
    disgust_precision = precision_score(y_true, y_pred, labels=[1], average='micro', zero_division=0)
    print(f"Class-specific metrics Score: {disgust_precision:.2%}")

    # Log Loss (Cross-Entropy Loss)
    loss = log_loss(y_true, y_pred_probs)
    print(f"Log-loss Score: {loss:.2%}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')  # Save to file
    plt.close()

    return test_acc, f1


def main():
    # Initialize everything
    train_gen, val_gen, test_gen = create_data_generators()
    model = build_resnet_model()

    print("\n=== Training dataset ===")
    print_class_distribution(train_dir)
    print("\n=== Testing dataset ===")
    print_class_distribution(test_dir)

    # Training pipeline
    trained_model = train_model(model, train_gen, val_gen)

    # Load best model and evaluate
    best_model = models.load_model('best_model.keras', custom_objects={'resnet_preprocess': resnet_preprocess})

    print("\n=== ResNet50 Layer Names ===")
    # Get the ResNet50 base model from your architecture
    base_model = best_model.get_layer("resnet50")
    for layer in base_model.layers[:5]:  # First 5 layers
        print(layer.name)

    visualize_conv_kernels(base_model, 'conv1_conv', -1, max_filters=32)
    visualize_conv_kernels(model, 'layer_name', layer_index=0, max_filters=-1)

    evaluate_model(best_model, test_gen)


if __name__ == "__main__":
    main()