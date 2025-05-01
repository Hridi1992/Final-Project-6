# ====================== IMPORTS AND CONFIGURATION ======================
# Core Python utilities
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

# Deep Learning framework
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input

# Machine Learning metrics
from sklearn.metrics import (
    f1_score, classification_report, cohen_kappa_score, log_loss,
    precision_score, top_k_accuracy_score, roc_auc_score,confusion_matrix,
    balanced_accuracy_score
)
from sklearn.utils import class_weight

# ====================== DATA CONFIGURATION ======================
# Dataset paths
train_dir = r'C:\Users\User\Documents\OsuSpring2025\DeepLearning\FProject\.venv\train'  # Contains class subfolders
test_dir = r'C:\Users\User\Documents\OsuSpring2025\DeepLearning\FProject\.venv\test'

# Image parameters
IMG_SIZE = 48 # Original image size from FER2013 dataset
TARGET_SIZE = 71 # Required input size for ResNet50
BATCH_SIZE = 32 # Number of samples processed before model update

# Emotion class labels
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

@register_keras_serializable()
def xception_preprocess(x):
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

def create_data_generators():
    """Create data pipelines with real-time augmentation"""
    # Training data augmentation configuration
    train_datagen = ImageDataGenerator(
        rescale=1./255, # Normalize pixel values
        rotation_range=15, # Random rotations ±15 degrees
        width_shift_range=0.15, # Horizontal shift ±20% of width
        height_shift_range=0.15, # Vertical shift ±20% of height
        brightness_range=[0.8, 1.2],
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True, # Random left-right flips
        fill_mode='constant',
        validation_split=0.2 # Holdout 20% for validation
    )

    # Test/validation data preprocessing (only normalization)
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Training data generator
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale', # Convert to single-channel
        batch_size=BATCH_SIZE,
        class_mode='categorical', # One-hot encoded labels
        classes=CLASS_NAMES,
        subset='training' # Training portion of split
    )

    # Validation data generator
    val_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_NAMES,
        shuffle=False, # Maintain order for evaluation
        subset='validation' # Validation portion
    )

    # Test data generator
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_NAMES,
        shuffle=False # Maintain original order
    )

    return train_generator, val_generator, test_generator


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

        # For grayscale images: squeeze channel dimension and use gray colormap
        plt.imshow(images[i].squeeze(), cmap='gray')

        # Add class label as title
        plt.title(CLASS_NAMES[class_indices[i]])
        plt.axis("off")

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

def transformer_block(x, num_heads=4, projection_dim=512, dropout_rate=0.1):
    """Transformer encoder block for attention-based feature learning"""
    # Multi-head self-attention mechanism
    attention_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=projection_dim // num_heads, # Dimension per attention head
        dropout=dropout_rate
    )(x, x) # Self-attention: query=key=value=x

    # First residual connection with layer normalization
    x = LayerNormalization(epsilon=1e-6)(x + attention_output)

    # Feed-forward network with GELU activation
    ffn = layers.Dense(projection_dim * 2, activation="gelu")(x) # Expand dimension
    ffn = layers.Dense(projection_dim)(ffn) # Compress back to original dimension
    ffn = layers.Dropout(dropout_rate)(ffn) # Regularization

    # Second residual connection with layer normalization
    return LayerNormalization(epsilon=1e-6)(x + ffn)


# ====================== MODEL ARCHITECTURE ======================
def build_mobilenet_model():
    """Build hybrid CNN-Transformer architecture"""
    # Input layer for grayscale images
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))

    # ========== DATA AUGMENTATION ==========
    # Real-time augmentation during training
    #x = layers.RandomFlip("horizontal")(inputs)
    #x = layers.RandomRotation(0.15)(x) # ±15% rotation
    #x = layers.RandomZoom(0.1)(x) # ±10% zoom
    #x = layers.RandomContrast(0.1)(x) # ±10% contrast variation

    # ========== PREPROCESSING ==========
    # Resize to ResNet input size
    x = layers.Resizing(TARGET_SIZE, TARGET_SIZE)(inputs)
    # Convert grayscale to RGB by repeating channels
    x = layers.Concatenate()([x, x, x])
    # Apply ResNet50 specific preprocessing
    x = tf.keras.applications.xception.preprocess_input(x)  # Official ResNet50 preprocessing

    # ========== BASE MODEL ==========
    # Initialize pre-trained ResNet50 without top layers
    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(TARGET_SIZE, TARGET_SIZE, 3)
    )
    x = base_model(x) # Pass through ResNet50

    print(f"Total layers in MobileNetV3: {len(base_model.layers)}")

    # ========== HYBRID ARCHITECTURE ==========
    # 1. CNN Feature Enhancement
    x = layers.Conv2D(1024, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x) # Reduce overfitting

    # 2. Transformer Preparation (Xception output shape is (7, 7, 2048))
    batch_size, height, width, channels = x.shape  # (None, 7, 7, 1024)
    x_reshape = layers.Reshape((height * width, channels))(x)  # (None, 49, 1024)

    # 3. Position Encoding
    positions = tf.range(start=0, limit=height * width, delta=1)
    position_embedding = layers.Embedding(
        input_dim=height * width,
        output_dim=channels
    )(positions)
    x_reshape += position_embedding

    # 4. Transformer Processing
    x_transformer = transformer_block(x_reshape, num_heads=8, projection_dim=1024)
    x_transformer = transformer_block(x_transformer, num_heads=8, projection_dim=1024)

    # 5. Feature Fusion
    x_transformer = layers.Reshape((height, width, channels))(x_transformer)
    x = layers.concatenate([x, x_transformer], axis=-1)
    # ========== HYBRID ARCHITECTURE CHANGES END HERE ==========

    # ========== CLASSIFICATION HEAD ==========
    # Feature refinement
    # Additional layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(7, activation='softmax')(x)

    model = models.Model(inputs,outputs)
    model.summary()

    return model


# ====================== TRAINING PIPELINE ======================
def train_model(model, train_gen, val_gen, initial_epochs=1, fine_tune_epochs=1):
    """Two-phase training process with transfer learning"""
    # Handle both initial training and fine-tuning phases"""

    # Phase 1: Frozen backbone (Initial training with frozen base)
    model.get_layer("xception").trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Early stopping to prevent overfitting
    initial_history = model.fit(
        train_gen,
        epochs=initial_epochs,
        validation_data=val_gen,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint('initial_best.keras', save_best_only=True)
        ]
    )

    # Phase 2: Fine-tuning
    print("\n=== Fine-Tuning ===")
    base_model = model.get_layer("xception")
    base_model.trainable = True

    # Unfreeze last 15% of layers
    num_layers = len(base_model.layers)
    for layer in base_model.layers[:int(num_layers * 0.5)]:
        layer.trainable = False

    # Exponential decay learning rate for stable fine-tuning
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-5,
        decay_steps=1000,
    )

    # Unfreeze last 35 layers (conv5_x blocks + top)
    #for layer in base_model.layers[140:]:
    #    layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    class_weight_dict = dict(enumerate(class_weights))

    fine_tune_history = model.fit(
        train_gen,
        epochs=initial_epochs + fine_tune_epochs,
        initial_epoch=initial_history.epoch[-1] + 1,
        validation_data=val_gen,
        class_weight = class_weight_dict,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True)
        ]
    )

    return model, initial_history, fine_tune_history


# ====================== VISUALIZATION & EVALUATION ======================
def visualize_conv_kernels(model, layer_name, layer_index, max_filters=32):
    """
    Visualize convolutional filters/weights for model interpretation

    Args:
        model: Keras model
        layer_name: Name of the Conv2D layer to visualize
        max_filters: Maximum number of filters to display (None for all)
    """
    if(max_filters == -1):
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
        inputs=model.input,  # Changed from model.inputs to model.input
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

def plot_resnet_feature_maps(model, input_image, layer_name=None, rows=4, cols=8):
    """
    Visualize feature maps from ResNet50's convolutional layers.
    """
    # 1. Get ResNet50 base model
    resnet = model.get_layer("xception")

    # 2. Get all convolutional layers from ResNet50
    resnet_conv_layers = [layer for layer in resnet.layers
                          if isinstance(layer, layers.Conv2D)]

    if not resnet_conv_layers:
        raise ValueError("No convolutional layers found in ResNet50")

    # 3. Recreate preprocessing pipeline
    processed_img = input_image
    if len(processed_img.shape) == 3:
        processed_img = np.expand_dims(processed_img, axis=0)

    # Apply your model's preprocessing steps manually
    processed_img = tf.image.resize(processed_img, [TARGET_SIZE, TARGET_SIZE])
    processed_img = tf.repeat(processed_img, 3, axis=-1)  # Grayscale to RGB
    processed_img = xception_preprocess(processed_img)  # ResNet50 preprocessing

    # 4. Create direct access to ResNet50
    resnet_input = resnet.input
    resnet_outputs = [layer.output for layer in resnet_conv_layers]

    # 5. Create feature map model
    feature_map_model = tf.keras.Model(
        inputs=resnet.input,
        outputs=resnet_outputs
    )

    # 6. Get all feature maps
    feature_maps = feature_map_model.predict(processed_img)

    # 7. Visualize each layer
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
    """Plot combined training/validation accuracy and loss across both phases (training progress)."""
    # Combine metrics from both training stages
    acc = initial_history.history['accuracy'] + fine_tune_history.history['accuracy']
    val_acc = initial_history.history['val_accuracy'] + fine_tune_history.history['val_accuracy']
    loss = initial_history.history['loss'] + fine_tune_history.history['loss']
    val_loss = initial_history.history['val_loss'] + fine_tune_history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
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
    # Get all test data from generator (Generate predictions)
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

    # Calculate metrics
    print("\n=== Final Evaluation ===")

    # Basic evaluation
    test_loss = log_loss(y_true, y_pred_probs)
    test_acc = balanced_accuracy_score(y_true, y_pred)

    print(f"Test Accuracy: {test_acc:.2%}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_true, y_pred):.2%}")
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f"Macro F1: {f1:.2%}")
    print(classification_report(y_true, y_pred,
                               target_names=CLASS_NAMES,
                               zero_division=0))

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

    # Confusion matrix visualization
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')  # Save to file
    plt.close()

    return test_acc, f1, cm_normalized


# ====================== MAIN EXECUTION ======================
def main():
    # Initialize data pipelines
    train_gen, val_gen, test_gen = create_data_generators()

    # Dataset analysis
    print("\n=== Training dataset ===")
    print_class_distribution(train_dir)
    print("\n=== Testing dataset ===")
    print_class_distribution(test_dir)

    # Visualize first 12 augmented images in 3x4 grid
    visualize_augmented_images(train_gen, num_samples=30, cols=6, figsize=(15, 8))

    # Model setup and training
    model = build_mobilenet_model()

    # Training pipeline
    trained_model, initial_history, fine_tune_history = train_model(model, train_gen, val_gen,
        initial_epochs=15,
        fine_tune_epochs=15)

    # Load best performing model
    best_model = models.load_model('best_model.keras',custom_objects={'resnet_preprocess': xception_preprocess})

    # Get the ResNet50 base model from your architecture
    print("\n=== MobileNetV3 Layer Names ===")
    base_model = best_model.get_layer("xception")
    for layer in base_model.layers[:5]:  # First 5 layers
        print(layer.name)

    visualize_conv_kernels(base_model, 'conv1_conv', -1, max_filters=128)
    visualize_conv_kernels(model, 'layer_name', layer_index=0, max_filters = -1)

    # Get sample image for feature map visualization
    test_gen.reset()
    sample_images, _ = next(test_gen)
    sample_image = sample_images[0]  # Get first image in batch

    # Visualize all ResNet50 conv layers
    plot_resnet_feature_maps(best_model, sample_image)
    # Plot feature maps from different convolutional layers
    #plot_feature_maps(model, sample_image, layer_index=0)  # First conv layer
    #plot_feature_maps(model, sample_image, layer_index=1)  # First conv layer

    # Training dynamics visualization
    plot_combined_history(initial_history, fine_tune_history)

    # Comprehensive evaluation
    test_acc, f1, cm = evaluate_model(best_model, test_gen)


if __name__ == "__main__":
    main()