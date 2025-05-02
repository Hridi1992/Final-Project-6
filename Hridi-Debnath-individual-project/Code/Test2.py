# ====================== IMPORTS AND CONFIGURATION ======================
# Core Python utilities
import os

os.environ['TF_KERAS_SAVE_FORMAT'] = 'keras'  # Force Keras v3 format
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduced TensorFlow warnings
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

# Deep Learning framework
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# Machine Learning metrics
from sklearn.metrics import (
    f1_score, classification_report, cohen_kappa_score, log_loss,
    precision_score, top_k_accuracy_score, roc_auc_score, confusion_matrix,
    balanced_accuracy_score
)
from sklearn.utils import class_weight
from sklearn.metrics import recall_score
from tensorflow.python.keras.models import save_model

MODEL_PATH = os.path.abspath('best_model.keras')

# ====================== DATA CONFIGURATION ======================
# Dataset paths
train_dir = r'/home/ubuntu/Project/archive/new/train'
test_dir = r'/home/ubuntu/Project/archive/new/test'

# Image parameters
IMG_SIZE = 48
TARGET_SIZE = 224
BATCH_SIZE = 32
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']


@register_keras_serializable()
def resnet_preprocess(x):
    """Custom preprocessing function for ResNet50 compatibility"""
    return preprocess_input(x)


def print_class_distribution(data_dir):
    """Analyze dataset balance by counting images per class"""
    print("\n=== Class Distribution ===")
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(data_dir, class_name)
        if os.path.exists(class_dir):
            num_images = len(os.listdir(class_dir))
            print(f"{class_name.capitalize():<9}: {num_images} images")
        else:
            print(f"{class_name.capitalize():<9}: Directory not found")


def visualize_augmented_images(generator, num_samples=8, cols=4, figsize=(15, 6)):
    """
    Visualizes augmented images from the data generator to inspect preprocessing effects.

    Parameters:
        generator (ImageDataGenerator): Data generator with augmentation
        num_samples (int): Number of augmented samples to display
        cols (int): Number of columns in the display grid
        figsize (tuple): Figure dimensions (width, height) in inches
    """
    # Get a batch of augmented data from the generator
    images, labels = next(generator)

    # Calculate grid dimensions
    rows = int(np.ceil(num_samples / cols))

    # Create figure and axes
    plt.figure(figsize=figsize)
    plt.suptitle("Augmented Training Images", y=1.05, fontsize=14)

    # Convert one-hot encoded labels back to class indices
    class_indices = np.argmax(labels[:num_samples], axis=1)

    # Plot each augmented image
    for i in range(num_samples):
        ax = plt.subplot(rows, cols, i + 1)

        # Reverse preprocessing and normalize for visualization
        img = images[i].squeeze()

        # Rescale from [-1, 1] to [0, 1]
        img = (img - img.min()) / (img.max() - img.min())

        plt.imshow(img, cmap='gray')
        plt.title(CLASS_NAMES[class_indices[i]])
        plt.axis("off")

    # Adjust layout and display
    plt.tight_layout()
    plt.show()


def create_data_generators():
    """Create data pipelines with real-time augmentation and validation"""
    # Training data augmentation
    train_datagen = ImageDataGenerator(
        preprocessing_function=resnet_preprocess,
        rotation_range=35,
        width_shift_range=0.25,
        height_shift_range=0.25,
        brightness_range=[0.6, 1.4],
        shear_range=0.4,
        zoom_range=0.4,
        horizontal_flip=True,
        channel_shift_range=50,
        fill_mode='constant',
        validation_split=0.2
    )

    # Test/validation data preprocessing
    test_datagen = ImageDataGenerator(preprocessing_function=resnet_preprocess)

    # Training data generator
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_NAMES,
        subset='training',
        seed=42
    )

    # Validation data generator
    val_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_NAMES,
        subset='validation',
        shuffle=False,
        seed=42
    )

    # Test data generator
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_NAMES,
        shuffle=False
    )

    # Verify generators
    print("\n=== Data Generators Verification ===")
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {val_generator.samples}")
    print(f"Test samples: {test_generator.samples}")

    # Test one batch
    x, y = next(train_generator)
    print(f"\nFirst training batch - X shape: {x.shape}, y shape: {y.shape}")
    print("Class distribution in batch:", np.unique(np.argmax(y, axis=1), return_counts=True))

    return train_generator, val_generator, test_generator


def build_resnet_model():
    """Build hybrid CNN-Transformer architecture"""
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))

    # Preprocessing
    x = layers.Resizing(TARGET_SIZE, TARGET_SIZE)(inputs)
    x = layers.Concatenate(axis=-1)([x, x, x])  # Proper grayscale to RGB conversion
    x = layers.Lambda(resnet_preprocess)(x)

    # Base ResNet50
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(TARGET_SIZE, TARGET_SIZE, 3),
        name='resnet50'
    )
    base_model.trainable = False  # Freeze initially

    # Channel attention
    def channel_attention(input_tensor):
        channel_axis = -1
        avg = layers.GlobalAveragePooling2D()(input_tensor)
        max = layers.GlobalMaxPooling2D()(input_tensor)
        avg = layers.Reshape((1, 1, avg.shape[1]))(avg)
        max = layers.Reshape((1, 1, max.shape[1]))(max)
        concat = layers.Concatenate(axis=channel_axis)([avg, max])
        conv = layers.Conv2D(1, (7, 7), padding='same', activation='sigmoid')(concat)
        return layers.Multiply()([input_tensor, conv])

    x = base_model(x)
    x = channel_attention(x)

    # Classification head
    x = layers.Conv2D(1024, (3, 3), activation='swish', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.6)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu',
                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(7, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.summary()
    return model


def train_model(model, train_gen, val_gen, initial_epochs=15, fine_tune_epochs=15):
    """Two-phase training process with transfer learning"""
    # Phase 1: Frozen backbone
    print("\n=== Initial Training ===")

    # Class weighting
    class_counts = np.bincount(train_gen.classes)
    median = np.median(class_counts)
    class_weights = {i: median / count for i, count in enumerate(class_counts)}
    print("Class weights:", class_weights)

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=15,
            monitor='val_accuracy',
            verbose=1,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            save_best_only=True,
            monitor='val_accuracy',
            mode='max'
        ),
        tf.keras.callbacks.CSVLogger('training_log.csv')
    ]

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-3),
        loss=tf.keras.losses.CategoricalFocalCrossentropy(alpha=0.25, gamma=2.0),
        metrics=['accuracy']
    )

    # Train
    print("\nStarting initial training...")
    initial_history = model.fit(
        train_gen,
        epochs=initial_epochs,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    print("Initial training history:", initial_history.history)

    # Phase 2: Fine-tuning
    print("\n=== Fine-Tuning ===")
    model.get_layer("resnet50").trainable = True
    for layer in model.get_layer("resnet50").layers[:100]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.CategoricalFocalCrossentropy(alpha=0.25, gamma=2.0),
        metrics=['accuracy']
    )

    fine_tune_history = model.fit(
        train_gen,
        initial_epoch=initial_history.epoch[-1] + 1,
        epochs=initial_epochs + fine_tune_epochs,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    print("Fine-tuning history:", fine_tune_history.history)

    return model, initial_history, fine_tune_history


def visualize_conv_kernels(model, layer_name, layer_index, max_filters=32):
    """Visualize convolutional filters/weights for model interpretation"""
    if (max_filters == -1):
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
        try:
            layer = model.get_layer(layer_name)
        except ValueError:
            print(f"Layer '{layer_name}' not found in the model.")
            return

        if not isinstance(layer, layers.Conv2D):
            print(f"Layer '{layer_name}' is not a Conv2D layer.")
            return

        kernels = layer.get_weights()[0]
        kernels = (kernels - kernels.min()) / (kernels.max() - kernels.min())

        num_filters = kernels.shape[-1]
        num_channels = kernels.shape[-2]

        if max_filters is not None:
            num_filters = min(num_filters, max_filters)

        cols = 8
        rows = int(np.ceil(num_filters / cols))

        fig = plt.figure(figsize=(cols * 2, rows * 2))
        plt.suptitle(f"Kernels from layer: {layer_name}", fontsize=16)

        for i in range(num_filters):
            ax = fig.add_subplot(rows, cols, i + 1)
            kernel = kernels[:, :, :, i]

            if kernel.shape[-1] == 3:
                ax.imshow(kernel)
            else:
                ax.imshow(np.mean(kernel, axis=-1), cmap='gray')

            ax.axis('off')

        plt.tight_layout()
        plt.show()


def plot_feature_maps(model, input_image, layer_index=0, rows=4, cols=8):
    """Plot the output feature maps from a specified convolutional layer"""
    conv_layers = [layer for layer in model.layers
                   if isinstance(layer, layers.Conv2D)]

    if not conv_layers:
        raise ValueError("No convolutional layers found in the model")

    if layer_index >= len(conv_layers):
        raise ValueError(f"Layer index {layer_index} exceeds number of conv layers ({len(conv_layers)})")

    activation_model = tf.keras.Model(
        inputs=model.input,
        outputs=conv_layers[layer_index].output
    )

    if len(input_image.shape) == 3:
        input_image = np.expand_dims(input_image, axis=0)

    activations = activation_model.predict(input_image, verbose=0)

    plt.figure(figsize=(cols * 2, rows * 2))

    for i in range(min(rows * cols, activations.shape[-1])):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(activations[0, :, :, i], cmap='viridis')
        plt.axis('off')
    plt.suptitle(f'Feature Maps from Layer {layer_index} ({conv_layers[layer_index].name})', y=0.92)
    plt.show()


def plot_resnet_feature_maps(model, input_image, layer_name=None, rows=4, cols=8):
    """Visualize feature maps from ResNet50's convolutional layers"""
    resnet = model.get_layer("resnet50")
    resnet_conv_layers = [layer for layer in resnet.layers
                          if isinstance(layer, layers.Conv2D)]

    if not resnet_conv_layers:
        raise ValueError("No convolutional layers found in ResNet50")

    processed_img = input_image
    if len(processed_img.shape) == 3:
        processed_img = np.expand_dims(processed_img, axis=0)

    processed_img = tf.image.resize(processed_img, [TARGET_SIZE, TARGET_SIZE])
    processed_img = tf.repeat(processed_img, 3, axis=-1)
    processed_img = resnet_preprocess(processed_img)

    resnet_input = resnet.input
    resnet_outputs = [layer.output for layer in resnet_conv_layers]

    feature_map_model = tf.keras.Model(
        inputs=resnet.input,
        outputs=resnet_outputs
    )

    feature_maps = feature_map_model.predict(processed_img)

    for layer_name, fmaps in zip([l.name for l in resnet_conv_layers], feature_maps):
        print(f"Visualizing layer: {layer_name}")

        n_filters = fmaps.shape[-1]
        plt.figure(figsize=(cols * 2, rows * 2))
        plt.suptitle(f"Layer: {layer_name}\nFilters: {n_filters}", y=0.95)

        for i in range(min(rows * cols, n_filters)):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(fmaps[0, :, :, i], cmap='viridis')
            plt.axis('off')

        plt.tight_layout()
        plt.show()


def plot_combined_history(initial_history, fine_tune_history):
    """Plot combined training/validation accuracy and loss across both phases"""
    acc = initial_history.history['accuracy'] + fine_tune_history.history['accuracy']
    val_acc = initial_history.history['val_accuracy'] + fine_tune_history.history['val_accuracy']
    loss = initial_history.history['loss'] + fine_tune_history.history['loss']
    val_loss = initial_history.history['val_loss'] + fine_tune_history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


def evaluate_model(model, test_gen):
    """Comprehensive model evaluation with multiple metrics"""
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

    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)

    print("\n=== Detailed Class Metrics ===")
    print(f"{'Class':<12} {'Recall':<8} {'Samples':<8}")
    for i, class_name in enumerate(CLASS_NAMES):
        class_samples = np.sum(y_true == i)
        print(f"{class_name.capitalize():<12} {recall_per_class[i]:<8.2%} {class_samples:<8}")

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=CLASS_NAMES,
        y=recall_per_class,
        hue=CLASS_NAMES,
        palette="viridis",
        legend=False
    )
    plt.title("Per-Class Recall Scores")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.ylabel("Recall")
    plt.xlabel("Emotion Class")
    plt.tight_layout()
    plt.savefig('recall_distribution.png')
    plt.close()

    test_loss = log_loss(y_true, y_pred_probs)
    test_acc = balanced_accuracy_score(y_true, y_pred)

    print(f"Test Accuracy: {test_acc:.2%}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_true, y_pred):.2%}")
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f"Macro F1: {f1:.2%}")
    print(classification_report(y_true, y_pred,
                                target_names=CLASS_NAMES,
                                zero_division=0))

    kappa = cohen_kappa_score(y_true, y_pred)
    print(f"Cohen Kappa Score: {kappa:.2%}")
    roc_auc = roc_auc_score(y_true, y_pred_probs, multi_class='ovr', average='macro')
    print(f"ROC AUC Score: {roc_auc:.2%}")
    top2_acc = top_k_accuracy_score(y_true, y_pred_probs, k=2)
    print(f"Top-k Score: {top2_acc:.2%}")
    disgust_precision = precision_score(y_true, y_pred, labels=[1], average='micro', zero_division=0)
    print(f"Class-specific metrics Score: {disgust_precision:.2%}")
    loss = log_loss(y_true, y_pred_probs)
    print(f"Log-loss Score: {loss:.2%}")

    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))

    with np.errstate(divide='ignore', invalid='ignore'):
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized, nan=0.0, posinf=0.0, neginf=0.0)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

    max_acc = np.max(cm_normalized.diagonal())
    max_acc = min(max_acc, 1.0)

    return max_acc


def main():
    # Initialize data pipelines
    train_gen, val_gen, test_gen = create_data_generators()

    # Dataset analysis
    print("\n=== Training dataset ===")
    print_class_distribution(train_dir)
    print("\n=== Testing dataset ===")
    print_class_distribution(test_dir)

    # Visualize augmented images
    visualize_augmented_images(train_gen, num_samples=30, cols=6, figsize=(15, 8))

    # Model setup and training
    model = build_resnet_model()

    # Training pipeline
    trained_model, initial_history, fine_tune_history = train_model(model, train_gen, val_gen,
                                                                    initial_epochs=15,
                                                                    fine_tune_epochs=15)

    # Load best model
    if os.path.exists(MODEL_PATH):
        print(f"\nLoading best model from {MODEL_PATH}")
        best_model = models.load_model(
            MODEL_PATH,
            custom_objects={'resnet_preprocess': resnet_preprocess}
        )
    else:
        print("\nNo saved model found. Using final trained model.")
        best_model = trained_model
        best_model.save(MODEL_PATH, save_format='keras')

    # Visualization
    print("\n=== ResNet50 Layer Names ===")
    base_model = best_model.get_layer("resnet50")
    for layer in base_model.layers[:5]:
        print(layer.name)

    visualize_conv_kernels(base_model, 'conv1_conv', -1, max_filters=128)
    visualize_conv_kernels(model, 'layer_name', layer_index=0, max_filters=-1)

    # Feature map visualization
    test_gen.reset()
    sample_images, _ = next(test_gen)
    sample_image = sample_images[0]

    plot_resnet_feature_maps(best_model, sample_image)
    plot_feature_maps(model, sample_image, layer_index=0)
    plot_feature_maps(model, sample_image, layer_index=1)

    # Training dynamics visualization
    plot_combined_history(initial_history, fine_tune_history)

    # Comprehensive evaluation
    max_acc = evaluate_model(best_model, test_gen)
    print(f"\nMaximum Class Accuracy: {max_acc:.2%}")


if __name__ == "__main__":
    main()
