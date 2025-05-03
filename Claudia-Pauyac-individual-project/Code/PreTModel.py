# ====================== IMPORTS AND CONFIGURATION ======================
import os
os.environ['TF_KERAS_SAVE_FORMAT'] = 'keras'  # Force Keras v3 format
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Core Python utilities
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
    precision_score, top_k_accuracy_score, roc_auc_score,confusion_matrix,
    balanced_accuracy_score
)
from sklearn.utils import class_weight
from sklearn.metrics import recall_score
from tensorflow.python.keras.models import save_model


MODEL_PATH = os.path.abspath('best_model.keras')
# ====================== DATA CONFIGURATION ======================
# Dataset paths
train_dir = r'C:\Users\User\Documents\OsuSpring2025\DeepLearning\FProject\.venv\train'  # Contains class subfolders
test_dir = r'C:\Users\User\Documents\OsuSpring2025\DeepLearning\FProject\.venv\test'

# Image parameters
IMG_SIZE = 48 # Original image size from FER2013 dataset
TARGET_SIZE = 224 # Required input size for ResNet50
BATCH_SIZE = 32 # Number of samples processed before model update

# Emotion class labels
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']


@register_keras_serializable(package="Custom", name="resnet_preprocess")
def resnet_preprocess(x):
    """Custom preprocessing function for ResNet50 compatibility"""
    return preprocess_input(x)


# ====================== DATA PREPARATION & ANALYSIS ======================
def print_class_distribution(data_dir):
    """
    Analyze and display class distribution by counting images per directory.

    Args:
        data_dir (str): Path to root directory containing class folders
    """
    print("\n=== Class Distribution ===")

    # Iterate through each emotion class
    for class_name in CLASS_NAMES:
        # Create full path to class directory
        class_dir = os.path.join(data_dir, class_name)

        # Check if directory exists before counting
        if os.path.exists(class_dir):
            # Count number of image files in directory
            num_images = len(os.listdir(class_dir))
            # Format output with aligned columns
            print(f"{class_name.capitalize():<9}: {num_images} images")
        else:
            # Handle missing directories gracefully
            print(f"{class_name.capitalize():<9}: Directory not found")


def create_data_generators():
    """
    Create data generators with augmentation for training/validation/test sets.

    Returns:
        tuple: (train_generator, val_generator, test_generator)
    """
    # Configure augmentation pipeline for training data
    train_datagen = ImageDataGenerator(
        preprocessing_function=resnet_preprocess,  # Model-specific preprocessing
        rotation_range=35,  # Random rotation ±35 degrees
        width_shift_range=0.25,  # Horizontal shift ±25% of width
        height_shift_range=0.25,  # Vertical shift ±25% of height
        brightness_range=[0.6, 1.4],  # Random brightness adjustment
        shear_range=0.4,  # Shear transformation intensity
        zoom_range=0.4,  # Random zoom range [60%, 140%]
        horizontal_flip=True,  # Random horizontal flips
        channel_shift_range=50,  # Random color channel shifts
        fill_mode='constant',  # Fill new pixels with constant value
        validation_split=0.2  # Reserve 20% for validation
    )

    # Configure preprocessing for test/validation data (no augmentation)
    test_datagen = ImageDataGenerator(
        preprocessing_function=resnet_preprocess  # Only preprocessing
    )

    # Training data generator with augmented samples
    train_generator = train_datagen.flow_from_directory(
        directory=train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),  # Resize images
        color_mode='rgb',  # Maintain 3 color channels
        batch_size=BATCH_SIZE,  # Samples per batch
        class_mode='categorical',  # One-hot encoded labels
        classes=CLASS_NAMES,  # Maintain class order
        subset='training',  # Use training portion of split
        interpolation='bicubic',  # High-quality resizing
        seed=42  # Reproducible randomness
    )

    # Validation data generator (uses same directory with subset)
    val_generator = train_datagen.flow_from_directory(
        directory=train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_NAMES,
        shuffle=False,  # Maintain order for evaluation
        subset='validation'  # Use validation portion
    )

    # Test data generator (separate directory)
    test_generator = test_datagen.flow_from_directory(
        directory=test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_NAMES,
        shuffle=False  # Maintain original test order
    )

    return train_generator, val_generator, test_generator


# ====================== DATA VISUALIZATION ======================
def visualize_augmented_images(generator, num_samples=8, cols=4, figsize=(15, 6),
                               output_dir='augmented_images'):
    """
    Visualize and save augmented training samples for quality control.

    Args:
        generator (ImageDataGenerator): Configured data generator
        num_samples (int): Number of images to display
        cols (int): Number of columns in grid layout
        figsize (tuple): Figure dimensions in inches
        output_dir (str): Directory to save visualization
    """
    # Get batch of augmented data from generator
    images, labels = next(generator)

    # Calculate grid layout dimensions
    rows = int(np.ceil(num_samples / cols))  # Dynamic row count

    # Create figure with title
    plt.figure(figsize=figsize)
    plt.suptitle("Augmented Training Images", y=1.05, fontsize=14)

    # Convert one-hot labels to class indices
    class_indices = np.argmax(labels[:num_samples], axis=1)

    # Plot each sample image
    for i in range(num_samples):
        ax = plt.subplot(rows, cols, i + 1)  # Create subplot

        # Process image for visualization
        img = images[i].squeeze()  # Remove batch dimension
        img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0,1]

        # Display image with class label
        plt.imshow(img, cmap='gray')
        plt.title(CLASS_NAMES[class_indices[i]])  # Add class name
        plt.axis("off")  # Remove axes

    # Final layout adjustments
    plt.tight_layout()

    # Save visualization to file
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "augmented_samples.png"))
    plt.close()  # Release memory resources


# ====================== MODEL ARCHITECTURE ======================
def transformer_block(x, num_heads=4, projection_dim=512, dropout_rate=0.1):
    """
    Transformer encoder block with self-attention and feed-forward network.

    Args:
        x (tensor): Input tensor
        num_heads (int): Number of attention heads
        projection_dim (int): Dimension of key/query vectors
        dropout_rate (float): Dropout probability

    Returns:
        tensor: Output tensor after transformer operations
    """
    # Multi-head self-attention mechanism
    attention_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=projection_dim // num_heads,  # Split dimension across heads
        dropout=dropout_rate
    )(x, x)  # Self-attention (q=k=v=x)

    # First residual connection + layer norm
    x = LayerNormalization(epsilon=1e-6)(x + attention_output)

    # Feed-forward network (FFN)
    ffn = layers.Dense(projection_dim * 2, activation="gelu")(x)  # Expand features
    ffn = layers.Dense(projection_dim)(ffn)  # Compress back
    ffn = layers.Dropout(dropout_rate)(ffn)  # Regularization

    # Second residual connection + layer norm
    return LayerNormalization(epsilon=1e-6)(x + ffn)


# ====================== MODEL ARCHITECTURE ======================
def build_resnet_model():
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
    x = layers.Concatenate(name="gray_to_rgb")([x, x, x])
    # Apply ResNet50 specific preprocessing
    x = layers.Lambda(
        resnet_preprocess,
        name="resnet_preprocess",
        output_shape=(TARGET_SIZE, TARGET_SIZE, 3)  # Explicit shape definition
    )(x)

    # ========== BASE MODEL ==========
    # Initialize pre-trained ResNet50 without top layers
    base_model = ResNet50(
        weights='imagenet', # Pre-trained on ImageNet
        include_top=False, # Exclude classification layers
        input_shape=(TARGET_SIZE, TARGET_SIZE, 3),
        name = 'resnet50'
    )

    # Add spatial attention before ResNet
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
     # Pass through ResNet50

    print(f"Total layers in ResNet50: {len(base_model.layers)}")

    # ========== HYBRID ARCHITECTURE ==========
    # 1. CNN Feature Enhancement
    x = layers.Conv2D(1024, (3, 3), activation='swish', padding='same')(x)
    x = layers.BatchNormalization(
        momentum=0.99,
        epsilon=1e-06,
        gamma_initializer='glorot_uniform'
    )(x)
    x = layers.Dropout(
        0.6,
        noise_shape=(None, 1, 1, x.shape[-1])  # Variational dropout
    )(x)

    # Global pooling and classification
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(7, activation='softmax',
                         kernel_initializer='lecun_normal')(x)

    model = models.Model(inputs, outputs)
    # Enable mixed precision training
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    model.summary()

    # 2. Transformer Preparation
    #batch_size, height, width, channels = x.shape  # (None, 7, 7, 512)
    #x_reshape = layers.Reshape((height * width, channels))(x)  # (None, 49, 512)

    # 3. Position Encoding
    #positions = tf.range(start=0, limit=height * width, delta=1)
    #position_embedding = layers.Embedding(
    #    input_dim=height * width,
    #    output_dim=channels
    #)(positions)  # (49, 512)
    #x_reshape += position_embedding  # Add positional information

    # 4. Transformer Processing
    #x_transformer = transformer_block(x_reshape, num_heads=8, projection_dim=512)
    #x_transformer = transformer_block(x_transformer, num_heads=8, projection_dim=512)

    # 5. Feature Fusion
    #x_transformer = layers.Reshape((height, width, channels))(x_transformer)
    # 6. Combine CNN and Transformer paths
    #x = layers.concatenate([x, x_transformer], axis=-1)  # Combine CNN+Transformer
    # ========== HYBRID ARCHITECTURE CHANGES END HERE ==========

    # ========== CLASSIFICATION HEAD ==========
    # Feature refinement
    #x = layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), kernel_regularizer=regularizers.L2(0.001), padding='same')(x)
    #x = layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    #x = layers.Activation('relu')(x)
    #x = layers.Dropout(0.5)(x)

    # Global pooling and dense layers
    #x = layers.GlobalAveragePooling2D()(x)
    #x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    #x = layers.Dropout(0.5)(x)
    #x = layers.Dense(256, activation='relu')(x)
    #x = layers.Dropout(0.5)(x)

    #x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.L2(0.001))(x)
    #x = layers.Dropout(0.5)(x)

    # Final classification layer
    #outputs = layers.Dense(7, activation='softmax')(x)

    return model


# ====================== TRAINING PIPELINE ======================
def train_model(model, train_gen, val_gen,
                initial_epochs=20,
                fine_tune_epochs=30,
                final_tune_epochs=10):
    """Two-phase training process with transfer learning"""
    # Handle both initial training and fine-tuning phases"""

    # Phase 1: Frozen backbone (Initial training with frozen base)
    print("\n=== Initial Training ===")
    model.get_layer("resnet50").trainable = False

    # Class weight calculation
    #class_weights = class_weight.compute_class_weight(
    #    'balanced',
    #    classes=np.unique(train_gen.classes),
    #    y=train_gen.classes
    #)
    #class_weight_dict = dict(enumerate(class_weights))

    # Class weighting
    class_counts = np.bincount(train_gen.classes)
    total = sum(class_counts)
    class_weights = {i: (1.0 / (count / total)) * 0.5 for i, count in enumerate(class_counts)}
    class_weights[1] *= 3.0  # Boost disgust class

    # Optimizer with cosine decay
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=3e-4,
        decay_steps=initial_epochs * len(train_gen)
    )

    # Phase 1: Frozen backbone
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(lr_schedule, weight_decay=1e-3),
        loss=tf.keras.losses.CategoricalFocalCrossentropy(alpha=0.25, gamma=2.0),
        metrics=['accuracy']
    )

    # Add learning rate reducer
    #lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
    #    monitor='val_loss',
    #    factor=0.5,
    #    patience=3,
    #    verbose=1
    #)

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            save_best_only=True,
            monitor='val_accuracy',
            mode='max'
        )
    ]

    # Early stopping to prevent overfitting
    initial_history = model.fit(
        train_gen,
        epochs=initial_epochs,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    # Phase 2: Fine-tuning
    print("\n=== Fine-Tuning ===")
    base_model = model.get_layer("resnet50")
    base_model.trainable = True

    # Freeze initial layers, unfreeze later layers
    #for layer in base_model.layers[:100]: # Freeze first 140 layers (80% of total)
    #    layer.trainable = False

    # Define unfreezing schedule (layer index ranges)
    unfreeze_schedule = [
        (160, 175, 10, 1e-5),  # Last 15 layers, 10 epochs, 1e-5 lr
        (140, 160, 15, 5e-6),  # Next 20 layers, 15 epochs, 5e-6 lr
        (100, 140, 20, 1e-6)  # Next 40 layers, 20 epochs, 1e-6 lr
    ]

    total_epochs = initial_epochs
    for start_idx, end_idx, epochs, lr in unfreeze_schedule:
        # Freeze all except current range
        for layer in base_model.layers:
            layer.trainable = False
        for layer in base_model.layers[start_idx:end_idx]:
            layer.trainable = True

        print(f"\nUnfreezing layers {start_idx}-{end_idx} (LR: {lr:.1e})")

        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr),
            loss=tf.keras.losses.CategoricalFocalCrossentropy(alpha=0.25, gamma=2.0),
            metrics=['accuracy']
        )

        history = model.fit(
            train_gen,
            initial_epoch=total_epochs,
            epochs=total_epochs + epochs,
            validation_data=val_gen,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        total_epochs += epochs

    print("\n=== Phase 3: Final Head Tuning ===")
    # Freeze entire base model
    base_model.trainable = False

    # Only unfreeze classification layers
    for layer in model.layers:
        if "dense" in layer.name or "dropout" in layer.name:
            layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-7),
        loss=tf.keras.losses.CategoricalFocalCrossentropy(alpha=0.25, gamma=2.0),
        metrics=['accuracy']
    )

    fine_tune_history = model.fit(
        train_gen,
        initial_epoch=total_epochs,
        epochs=total_epochs + final_tune_epochs,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    return model, initial_history, fine_tune_history


# ====================== VISUALIZATION & EVALUATION ======================
def visualize_conv_kernels(model, layer_name, layer_index, max_filters=32, output_dir='conv_kernels'):
    """
    Visualize and save convolutional filters/weights for model interpretation.

    Args:
        model (keras.Model): Trained Keras model
        layer_name (str): Name of the Conv2D layer to visualize
        layer_index (int): Index of convolutional layer (when max_filters=-1)
        max_filters (int): Maximum filters to display (-1 for legacy mode)
        output_dir (str): Output directory for saved images
    """
    # Legacy mode: Visualize using layer index
    if max_filters == -1:
        # Get all Conv2D layers using isinstance for accurate type checking
        conv_layers = [layer for layer in model.layers
                       if isinstance(layer, layers.Conv2D)]

        # Validate convolutional layers exist
        if not conv_layers:
            raise ValueError("No convolutional layers found in the model")

        # Get specified layer and its weights
        layer = conv_layers[layer_index]
        kernels, _ = layer.get_weights()

        # Normalize kernel values for visualization [0, 1]
        kernels = (kernels - kernels.min()) / (kernels.max() - kernels.min())

        # Calculate grid dimensions
        n_filters = kernels.shape[-1]  # Number of filters in layer
        n_cols = 8  # Fixed number of columns
        n_rows = int(np.ceil(n_filters / n_cols))  # Dynamic rows

        # Create figure and plot kernels
        plt.figure(figsize=(n_cols * 2, n_rows * 2))
        for i in range(n_filters):
            plt.subplot(n_rows, n_cols, i + 1)
            plt.imshow(kernels[:, :, 0, i], cmap='viridis')  # First channel only
            plt.axis('off')

        # Add title and layout adjustments
        plt.suptitle(f'Convolutional Kernels from Layer {layer_index}', y=0.95)
        plt.tight_layout()

        # Save visualization
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{layer.name}_kernels.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

    # Modern mode: Visualize using layer name
    else:
        # Retrieve layer by name with error handling
        try:
            layer = model.get_layer(layer_name)
        except ValueError:
            print(f"Layer '{layer_name}' not found in the model.")
            return

        # Validate layer type
        if not isinstance(layer, layers.Conv2D):
            print(f"Layer '{layer_name}' is not a Conv2D layer.")
            return

        # Extract and normalize kernels
        kernels = layer.get_weights()[0]
        kernels = (kernels - kernels.min()) / (kernels.max() - kernels.min())

        # Calculate grid dimensions
        num_filters = min(kernels.shape[-1], max_filters)
        cols = 8
        rows = int(np.ceil(num_filters / cols))

        # Create figure and plot kernels
        fig = plt.figure(figsize=(cols * 2, rows * 2))
        plt.suptitle(f"Kernels from layer: {layer.name}", fontsize=16)

        for i in range(num_filters):
            ax = fig.add_subplot(rows, cols, i + 1)
            kernel = kernels[:, :, :, i]

            # Handle multi-channel vs single-channel kernels
            if kernel.shape[-1] == 3:
                ax.imshow(kernel)  # RGB visualization
            else:
                ax.imshow(np.mean(kernel, axis=-1), cmap='gray')  # Grayscale

            ax.axis('off')

        # Save visualization
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{layer.name}_kernels.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()


def plot_feature_maps(model, input_image, layer_index=0, rows=4, cols=8,
                      output_dir='feature_maps', filename=None):
    """
    Visualize and save feature maps from specified convolutional layer.

    Args:
        model (keras.Model): Trained Keras model
        input_image (np.array): Preprocessed input image (H, W, C)
        layer_index (int): Index of conv layer to visualize
        rows (int): Grid rows for display
        cols (int): Grid columns for display
        output_dir (str): Output directory path
        filename (str): Optional custom filename
    """
    # Get all Conv2D layers in model
    conv_layers = [layer for layer in model.layers
                   if isinstance(layer, layers.Conv2D)]

    # Validate layer existence
    if not conv_layers:
        raise ValueError("No convolutional layers found in the model")

    if layer_index >= len(conv_layers):
        raise ValueError(f"Invalid layer index {layer_index} for {len(conv_layers)} conv layers")

    # Create activation submodel
    activation_model = tf.keras.Model(
        inputs=model.input,
        outputs=conv_layers[layer_index].output
    )

    # Add batch dimension if missing
    if len(input_image.shape) == 3:
        input_image = np.expand_dims(input_image, axis=0)

    # Generate activations
    activations = activation_model.predict(input_image, verbose=0)

    # Create visualization figure
    n_filters = activations.shape[-1]
    plt.figure(figsize=(cols * 2, rows * 2))

    # Plot feature maps
    for i in range(min(rows * cols, n_filters)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(activations[0, :, :, i], cmap='viridis')
        plt.axis('off')

    # Add titles and layout
    layer = conv_layers[layer_index]
    plt.suptitle(f'Feature Maps: Layer {layer_index} ({layer.name})', y=0.92)
    plt.tight_layout()

    # Generate filename if not provided
    if filename is None:
        filename = f"layer_{layer_index}_{layer.name}_feature_maps.png"

    # Save visualization
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
    plt.close()


def plot_resnet_feature_maps(model, input_image, rows=4, cols=8,
                             output_dir='resnet_feature_maps', filename_prefix='',
                             target_size=224):
    """
    Visualize and save ResNet50 feature maps with proper preprocessing.

    Args:
        model (keras.Model): Model containing ResNet50 base
        input_image (np.array): Input image array
        rows (int): Grid rows per layer
        cols (int): Grid columns per layer
        output_dir (str): Output directory path
        filename_prefix (str): Filename prefix for saved images
        target_size (int): Input size for ResNet preprocessing
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Extract ResNet50 base model
    resnet = model.get_layer("resnet50")

    # Get all Conv2D layers
    resnet_conv_layers = [layer for layer in resnet.layers
                          if isinstance(layer, layers.Conv2D)]

    # Validate layers
    if not resnet_conv_layers:
        raise ValueError("No Conv2D layers found in ResNet50")

    # Preprocess input image
    processed_img = input_image
    if len(processed_img.shape) == 3:
        processed_img = np.expand_dims(processed_img, axis=0)

    # Resize and convert to RGB
    processed_img = tf.image.resize(processed_img, [target_size, target_size])
    processed_img = tf.repeat(processed_img, 3, axis=-1)  # Handle grayscale
    processed_img = tf.keras.applications.resnet50.preprocess_input(processed_img)

    # Create feature map model
    feature_map_model = tf.keras.Model(
        inputs=resnet.input,
        outputs=[layer.output for layer in resnet_conv_layers]
    )

    # Generate all feature maps
    feature_maps = feature_map_model.predict(processed_img)

    # Visualize each layer's feature maps
    for layer, fmaps in zip(resnet_conv_layers, feature_maps):
        print(f"Processing layer: {layer.name}")

        # Create figure
        plt.figure(figsize=(cols * 2, rows * 2))
        plt.suptitle(f"Feature Maps: {layer.name}\nTotal Filters: {fmaps.shape[-1]}", y=0.95)

        # Plot first n filters
        n_to_plot = min(rows * cols, fmaps.shape[-1])
        for i in range(n_to_plot):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(fmaps[0, :, :, i], cmap='viridis')
            plt.axis('off')

        # Save and close
        plt.tight_layout()
        filename = f"{filename_prefix}{layer.name}_feature_maps.png"
        plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
        plt.close()


def plot_combined_history(initial_history, fine_tune_history, output_dir='training_history'):
    """
    Visualize and save combined training history from two phases.

    Args:
        initial_history: History object from initial training
        fine_tune_history: History object from fine-tuning
        output_dir (str): Output directory path
    """
    # Combine metrics from both phases
    acc = initial_history.history['accuracy'] + fine_tune_history.history['accuracy']
    val_acc = initial_history.history['val_accuracy'] + fine_tune_history.history['val_accuracy']
    loss = initial_history.history['loss'] + fine_tune_history.history['loss']
    val_loss = initial_history.history['val_loss'] + fine_tune_history.history['val_loss']

    # Create epoch range
    epochs = range(1, len(acc) + 1)

    # Create figure
    plt.figure(figsize=(14, 5))

    # Accuracy subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Save and close
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "training_history.png"))
    plt.close()

    print(f"Training history saved to {output_dir}/training_history.png")


def evaluate_model(model, test_gen, class_names, output_dir='evaluation_results'):
    """
    Comprehensive model evaluation with visualizations.

    Args:
        model: Trained Keras model
        test_gen: Test data generator
        class_names: List of class labels
        output_dir: Output directory path
    Returns:
        float: Maximum class accuracy
    """
    # Generate predictions
    test_gen.reset()
    y_true = []
    y_pred_probs = []

    for _ in range(len(test_gen)):
        x, y = next(test_gen)
        y_true.extend(np.argmax(y, axis=1))
        y_pred_probs.extend(model.predict(x, verbose=0))

    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred_probs = np.vstack(y_pred_probs)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Calculate metrics
    print("\n=== Final Evaluation Metrics ===")

    # Per-class recall
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)

    # Print class metrics
    print(f"\n{'Class':<12} {'Recall':<8} {'Samples':<8}")
    for i, name in enumerate(class_names):
        print(f"{name.capitalize():<12} {recall_per_class[i]:<8.2%} {(y_true == i).sum():<8}")

    # Recall visualization
    plt.figure(figsize=(10, 6))
    sns.barplot(x=class_names, y=recall_per_class, palette="viridis")
    plt.title("Per-Class Recall Scores")
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.ylabel("Recall")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "recall_distribution.png"))
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized, nan=0.0)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    # Calculate additional metrics
    test_loss = log_loss(y_true, y_pred_probs)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred_probs, multi_class='ovr', average='macro')

    # Print metrics
    print(f"\nBalanced Accuracy: {balanced_acc:.2%}")
    print(f"Macro F1: {f1:.2%}")
    print(f"Cohen's Kappa: {kappa:.2%}")
    print(f"ROC AUC: {roc_auc:.2%}")
    print(f"Log Loss: {test_loss:.4}")

    # Return maximum class accuracy
    return np.max(np.diag(cm_normalized))


# ====================== MAIN EXECUTION PIPELINE ======================
def main():
    """End-to-end execution pipeline for model training and evaluation."""

    # ====================== DATA PREPARATION ======================
    # Initialize data generators for all splits
    train_gen, val_gen, test_gen = create_data_generators()  # Returns tuple of generators

    # ====================== DATASET ANALYSIS ======================
    # Display class distribution statistics
    print("\n=== Training Dataset Analysis ===")
    print_class_distribution(train_dir)  # Analyze training data balance

    print("\n=== Testing Dataset Analysis ===")
    print_class_distribution(test_dir)  # Analyze test set composition

    # ====================== DATA QUALITY CHECKS ======================
    # Visualize augmented samples for preprocessing verification
    visualize_augmented_images(
        generator=train_gen,
        num_samples=30,  # Generate 30 sample images
        cols=6,  # 6-column grid layout
        figsize=(15, 8),  # Large figure for clarity
        output_dir='augmented_images'  # Save to dedicated folder
    )

    # ====================== MODEL SETUP ======================
    # Construct model architecture
    model = build_resnet_model()  # Custom ResNet-based architecture

    # ====================== TRAINING PHASES ======================
    # Execute multi-phase training process
    trained_model, initial_history, fine_tune_history = train_model(
        model=model,
        train_gen=train_gen,
        val_gen=val_gen,
        initial_epochs=1,  # Feature extraction phase
        fine_tune_epochs=1,  # Fine-tuning phase
        final_tune_epochs=1  # Final optimization
    )

    # ====================== MODEL MANAGEMENT ======================
    # Model loading with existence check
    if os.path.exists(MODEL_PATH):
        print(f"\nLoading best model from {MODEL_PATH}")
        best_model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={'resnet_preprocess': resnet_preprocess}  # Handle custom layer
        )
    else:
        print("\nNo saved model found. Using final trained model.")
        best_model = trained_model  # Fallback to latest version
        best_model.save(MODEL_PATH, save_format='keras')  # Persist model

    # ====================== MODEL INSPECTION ======================
    # Verify base model layer structure
    print("\n=== ResNet50 Layer Names (First 5 Layers) ===")
    base_model = best_model.get_layer("resnet50")  # Access base architecture
    for layer in base_model.layers[:5]:  # Inspect initial layers
        print(layer.name)  # Display layer names for verification

    # ====================== MODEL INTERPRETATION ======================
    # Visualize convolutional filters
    visualize_conv_kernels(
        model=base_model,
        layer_name='conv1_conv',  # Specific ResNet layer
        layer_index=-1,  # Legacy mode indicator
        max_filters=128,  # Limit displayed filters
        output_dir='kernels'  # Save to kernels directory
    )

    # Alternative visualization method
    visualize_conv_kernels(
        model=model,
        layer_name='layer_name',  # Should be replaced with actual layer name
        layer_index=0,  # First convolutional layer
        max_filters=-1,  # Show all filters
        output_dir='kernels'
    )

    # ====================== FEATURE ANALYSIS ======================
    # Prepare sample image for visualization
    test_gen.reset()  # Reset generator to start
    sample_images, _ = next(test_gen)  # Get batch from test set
    sample_image = sample_images[0]  # Extract first image

    # Generate ResNet feature map visualizations
    plot_resnet_feature_maps(
        model=best_model,
        input_image=sample_image,
        output_dir='featureMaps',  # Unified output directory
        filename_prefix='exp1_'  # Experiment identifier
    )

    # Visualize specific convolutional layers
    plot_feature_maps(  # First convolutional layer
        model=model,
        input_image=sample_image,
        layer_index=0,
        output_dir='featureMaps'
    )
    plot_feature_maps(  # Second convolutional layer
        model=model,
        input_image=sample_image,
        layer_index=1,
        output_dir='featureMaps'
    )

    # ====================== TRAINING ANALYSIS ======================
    # Visualize training progress across phases
    plot_combined_history(
        initial_history=initial_history,  # Feature extraction phase
        fine_tune_history=fine_tune_history,  # Fine-tuning phase
        output_dir='results'  # Save visualization to results/
    )

    # ====================== FINAL EVALUATION ======================
    # Comprehensive model performance assessment
    max_acc = evaluate_model(
        model=best_model,
        test_gen=test_gen  # Use held-out test set
    )
    print(f"\nMaximum Class Accuracy: {max_acc:.2%}")  # Display peak performance


if __name__ == "__main__":
    main()