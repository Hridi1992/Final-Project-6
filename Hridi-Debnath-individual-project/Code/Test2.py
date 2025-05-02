# ====================== IMPORTS AND CONFIGURATION ======================
# Core Python utilities
# Add at the very top 🔧
import os
os.environ['TF_KERAS_SAVE_FORMAT'] = 'keras'  # Force Keras v3 format
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
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
    precision_score, top_k_accuracy_score, roc_auc_score,confusion_matrix,
    balanced_accuracy_score
)
from sklearn.utils import class_weight
from sklearn.metrics import recall_score
from tensorflow.python.keras.models import save_model

MODEL_PATH = os.path.abspath('best_model.keras')
# ====================== DATA CONFIGURATION ======================
# Dataset paths
train_dir = r'/home/ubuntu/Project/archive/new/train'  # Contains class subfolders
test_dir = r'/home/ubuntu/Project/archive/new/test'

# Image parameters
IMG_SIZE = 48 # Original image size from FER2013 dataset
TARGET_SIZE = 224 # Required input size for ResNet50
BATCH_SIZE = 32 # Number of samples processed before model update

# Emotion class labels
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

def create_data_generators():
    """Create data pipelines with real-time augmentation"""
    # Training data augmentation configuration
    train_datagen = ImageDataGenerator(
        preprocessing_function=resnet_preprocess,
        rotation_range=35, # Random rotations ±15 degrees
        width_shift_range=0.25, # Horizontal shift ±20% of width
        height_shift_range=0.25, # Vertical shift ±20% of height
        brightness_range=[0.6, 1.4],
        shear_range=0.4,
        zoom_range=0.4,
        horizontal_flip=True, # Random left-right flips
        channel_shift_range=50,
        fill_mode='constant',
        validation_split=0.2 # Holdout 20% for validation
    )

    # Test/validation data preprocessing (only normalization)
    test_datagen = ImageDataGenerator(preprocessing_function=resnet_preprocess)

    # Training data generator
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='rgb', # Convert to single-channel
        batch_size=BATCH_SIZE,
        class_mode='categorical', # One-hot encoded labels
        classes=CLASS_NAMES,
        subset='training',
        interpolation='bicubic',
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
        shuffle=False, # Maintain order for evaluation
        subset='validation' # Validation portion
    )

    # Test data generator
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='rgb',
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
    x = layers.Concatenate()([x, x, x])
    # Apply ResNet50 specific preprocessing
    x = layers.Lambda(resnet_preprocess)(x)  # Official ResNet50 preprocessing

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
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.6)(x) # Reduce overfitting

    # Global pooling and classification
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(7, activation='softmax',
                         kernel_initializer='lecun_normal')(x)

    model = models.Model(inputs, outputs)
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
    resnet = model.get_layer("resnet50")

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
    processed_img = resnet_preprocess(processed_img)  # ResNet50 preprocessing

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

    # Per-class recall calculation
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)

    print("\n=== Detailed Class Metrics ===")
    print(f"{'Class':<12} {'Recall':<8} {'Samples':<8}")
    for i, class_name in enumerate(CLASS_NAMES):
        class_samples = np.sum(y_true == i)
        print(f"{class_name.capitalize():<12} {recall_per_class[i]:<8.2%} {class_samples:<8}")

    # Visualize recall distribution
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=CLASS_NAMES,
        y=recall_per_class,
        hue=CLASS_NAMES,  # Add hue mapping
        palette="viridis",
        legend=False  # Disable redundant legend
    )
    plt.title("Per-Class Recall Scores")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.ylabel("Recall")
    plt.xlabel("Emotion Class")
    plt.tight_layout()
    plt.savefig('recall_distribution.png')
    plt.close()


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

    # Handle potential division by zero
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
    plt.savefig('confusion_matrix.png')  # Save to file
    plt.close()

    # Get maximum valid class accuracy
    max_acc = np.max(cm_normalized.diagonal())
    max_acc = min(max_acc, 1.0)  # Ensure never exceeds 100%

    return max_acc


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
    model = build_resnet_model()

    # Training pipeline
    trained_model, initial_history, fine_tune_history = train_model(model,
        train_gen,
        val_gen,
        initial_epochs=15,
        fine_tune_epochs=15,  # Sum of all progressive stage epochs
        final_tune_epochs=15
    )

    # 🔧 Revised model loading with existence check
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

    # Load best performing model
    #best_model = models.load_model('best_model.keras',custom_objects={'resnet_preprocess': resnet_preprocess})

    # Get the ResNet50 base model from your architecture
    print("\n=== ResNet50 Layer Names ===")
    base_model = best_model.get_layer("resnet50")
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
    plot_feature_maps(model, sample_image, layer_index=0)  # First conv layer
    plot_feature_maps(model, sample_image, layer_index=1)  # First conv layer

    # Training dynamics visualization
    plot_combined_history(initial_history, fine_tune_history)

    # Comprehensive evaluation
    max_acc = evaluate_model(best_model, test_gen)
    print(f"\nMaximum Class Accuracy: {max_acc:.2%}")


if __name__ == "__main__":
    main()