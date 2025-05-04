# ====================== IMPORTS AND CONFIGURATION ======================
# Core Python utilities
import os
os.environ['TF_KERAS_SAVE_FORMAT'] = 'keras'  # Force Keras v3 format for model saving
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow info messages

import numpy as np  # Numerical computing
import seaborn as sns  # Statistical visualization
import matplotlib.pyplot as plt  # Plotting
from collections import defaultdict  # Dictionary with default values

# Deep Learning framework
from tensorflow.keras import layers, models, regularizers  # Core Keras components
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization  # Transformer components
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Data augmentation
from tensorflow.keras.saving import register_keras_serializable  # Custom layer serialization
from tensorflow.keras.applications import ResNet50  # Pre-trained CNN backbone
from tensorflow.keras.applications.resnet50 import preprocess_input  # ResNet-specific preprocessing
from sklearn.utils.class_weight import compute_class_weight  # Class imbalance handling

# Machine Learning metrics
from sklearn.metrics import (
    f1_score, classification_report, cohen_kappa_score, log_loss,
    precision_score, top_k_accuracy_score, roc_auc_score, confusion_matrix,
    balanced_accuracy_score
)
from sklearn.utils import class_weight
from sklearn.metrics import recall_score
from tensorflow.python.keras.models import save_model  # Model saving

MODEL_PATH = os.path.abspath('best_model.keras')  # Path for saving best model

# ====================== DATA CONFIGURATION ======================
# Dataset paths
train_dir = r'/home/ubuntu/FinalProject/.venv/train'  # Training data directory
test_dir = r'/home/ubuntu/FinalProject/.venv/test'  # Test data directory

# Image parameters
IMG_SIZE = 48  # Original image size from dataset
TARGET_SIZE = 224  # ResNet50 required input size
BATCH_SIZE = 32  # Number of samples per training batch

# Emotion class labels
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Custom preprocessing registration for model serialization
@register_keras_serializable(package="Custom", name="resnet_preprocess")
def resnet_preprocess(x):
    """ResNet50-specific preprocessing (channel-wise center normalization)"""
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
    """Create data generators with augmentation for training/validation/test sets"""
    # Training data augmentation configuration
    train_datagen = ImageDataGenerator(
        rotation_range=35,  # Random rotation ±35 degrees
        width_shift_range=0.25,  # Horizontal shift up to 25% width
        height_shift_range=0.25,  # Vertical shift up to 25% height
        brightness_range=[0.5, 1.5],  # Random brightness adjustment
        shear_range=0.4,  # Shear transformation intensity
        zoom_range=0.4,  # Random zoom between 60%-140%
        horizontal_flip=True,  # Random horizontal flips
        fill_mode='constant',  # Fill new pixels with constant value
        validation_split=0.2  # 20% validation split
    )

    # Test/validation preprocessing (no augmentation)
    test_datagen = ImageDataGenerator()

    # Training generator with augmentation
    train_generator = train_datagen.flow_from_directory(
        directory=train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',  # Input as single-channel
        batch_size=BATCH_SIZE,
        class_mode='categorical',  # One-hot encoded labels
        subset='training',  # Training portion of split
        seed=42  # Reproducible randomness
    )

    # Validation generator
    val_generator = train_datagen.flow_from_directory(
        directory=train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,  # Maintain order for evaluation
        subset='validation'  # Validation portion
    )

    # Test generator
    test_generator = test_datagen.flow_from_directory(
        directory=test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False  # Maintain original order
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
def efficient_transformer_block(x, num_heads=4, projection_dim=None, dropout=0.1):
    """Transformer block with efficient attention and feed-forward network"""
    input_channels = x.shape[-1]
    projection_dim = projection_dim or input_channels

    # Multi-head self-attention
    attn = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=projection_dim // num_heads,
        dropout=dropout
    )(x, x)

    # Channel adjustment if needed
    if attn.shape[-1] != input_channels:
        attn = layers.Conv2D(input_channels, 1)(attn)

    # Add & Norm
    x = LayerNormalization(epsilon=1e-6)(x + attn)

    # Feed-forward network with GELU activation
    ffn = layers.Conv2D(projection_dim * 4, 1, activation='gelu')(x)
    ffn = layers.Conv2D(input_channels, 1)(ffn)
    ffn = layers.Dropout(dropout)(ffn)

    return LayerNormalization(epsilon=1e-6)(x + ffn)


def spatial_channel_attention(input_tensor, reduction_ratio=8):
    """Dual attention mechanism for both channel and spatial refinement"""
    # ===== Channel Attention =====
    channel = input_tensor.shape[-1]

    # Global context learning
    avg_pool = layers.GlobalAveragePooling2D(keepdims=True)(input_tensor)
    max_pool = layers.GlobalMaxPooling2D(keepdims=True)(input_tensor)

    # Shared MLP for channel weighting
    mlp = layers.Dense(channel // reduction_ratio, activation='relu')
    avg_out = mlp(avg_pool)
    max_out = mlp(max_pool)

    # Channel attention weights
    channel_weights = layers.Add()([avg_out, max_out])
    channel_weights = layers.Dense(channel, activation='sigmoid')(channel_weights)

    # ===== Spatial Attention =====
    # Concatenate pooled features
    avg_spatial = layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(input_tensor)
    max_spatial = layers.Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(input_tensor)
    spatial_concat = layers.Concatenate(axis=-1)([avg_spatial, max_spatial])
    # Spatial attention weights
    spatial_weights = layers.Conv2D(1, (7, 7), padding='same', activation='sigmoid')(spatial_concat)

    # ===== Combine Both Attentions =====
    # Apply channel attention first
    refined = layers.Multiply()([input_tensor, channel_weights])
    # Then apply spatial attention
    refined = layers.Multiply()([refined, spatial_weights])

    # Residual connection
    return layers.Add()([input_tensor, refined])


def build_class_branch(x, emotion):
    """Class-specific channel weighting without spatial redundancy"""
    # Emotion-specific channel attention
    channel_weights = layers.Dense(
        x.shape[-1],
        activation='sigmoid',
        name=f"{emotion}_channel_weights"
    )(layers.GlobalAvgPool2D()(x))

    # Apply channel weights
    weighted_features = layers.Multiply()([x, channel_weights])

    # Feature aggregation
    x_out = layers.GlobalAvgPool2D()(weighted_features)
    x_out = layers.Dense(128, activation='gelu')(x_out)

    return x_out


def build_resnet_model():
    """Build hybrid ResNet50-Transformer architecture with attention mechanisms"""
    # Input layer for grayscale images
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))

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
        pooling=None,
        name = 'resnet50'
    )

    # Strategic layer unfreezing
    for layer in base_model.layers[:100]:
        layer.trainable = False

    x = base_model(x)

    # ========== DUAL ATTENTION BLOCK ==========
    x = spatial_channel_attention(x) # Add this line
    # Transformer-enhanced feature refinement
    x = efficient_transformer_block(x,num_heads=8)

    def build_class_branch(x, emotion):
        """Specialized branch for challenging emotions"""
        # Emotion-specific features
        x_out = layers.GlobalAvgPool2D()(x)
        x_out = layers.Dense(128, activation='gelu')(x_out)
        return x_out

    # Class-specific attention branches
    main_branch = layers.GlobalAvgPool2D()(x)
    angry_branch = build_class_branch(x, 'angry')
    fear_branch = build_class_branch(x, 'fear')
    disgust_branch = build_class_branch(x, 'disgust')
    sad_branch = build_class_branch(x, 'sad')

    # Fusion layer
    combined = layers.Concatenate()([main_branch, angry_branch, disgust_branch, fear_branch, sad_branch])
    combined = layers.Dense(512, activation='gelu')(combined)
    combined = layers.Dropout(0.5)(combined)
    # Output
    outputs = layers.Dense(7, activation='softmax')(combined)

    print(f"Total layers in ResNet50: {len(base_model.layers)}")

    model = models.Model(inputs, outputs)

    # Enable mixed precision training
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    model.summary()

    return model


# ====================== TRAINING PIPELINE ======================
def train_model(model, train_gen, val_gen,
                initial_epochs=20,
                fine_tune_epochs=30,
                final_tune_epochs=10):
    """Three-phase training process with progressive unfreezing"""
    # Class weighting for imbalanced data
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    class_weights_dict = {i: w for i, w in enumerate(class_weights)}

    # Common callbacks
    base_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=15,
            monitor='val_loss',
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
    ]

    full_callbacks = base_callbacks + [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
    ]

    # Training phase 1: Frozen base model
    base_model = model.get_layer("resnet50")
    base_model.trainable = False

    # Custom learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-5,
        decay_steps=initial_epochs * len(train_gen),
        alpha=3e-4
    )

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(lr_schedule, weight_decay=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Recall(class_id=1)]
    )

    print("\n=== Phase 1: Initial Training ===")
    initial_history = model.fit(
        train_gen,
        initial_epoch=0,  # Start from epoch 0
        epochs=initial_epochs,
        validation_data=val_gen,
        class_weight=class_weights_dict,
        callbacks=base_callbacks,
        verbose=1
    )

    # Collect all histories
    all_histories = [initial_history]
    current_epoch = initial_epochs

    # Training phase 2: Progressive unfreezing
    print("\n=== Phase 2: Fine-Tuning ===")
    unfreeze_schedule = [
        (160, 175, 10, 1e-5),  # Last layers
        (140, 160, 15, 5e-6),  # Middle layers
        (100, 140, 20, 1e-6)  # Earlier layers
    ]

    for start_idx, end_idx, epochs, lr in unfreeze_schedule:
        # Freeze all except current range
        base_model.trainable = False
        for layer in base_model.layers[start_idx:end_idx]:
            layer.trainable = True

        print(f"\nUnfreezing layers {start_idx}-{end_idx} (LR: {lr:.1e})")

        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        phase_history = model.fit(
            train_gen,
            initial_epoch=current_epoch,
            epochs=current_epoch + epochs,
            validation_data=val_gen,
            class_weight=class_weights_dict,
            callbacks=full_callbacks,
            verbose=1
        )
        all_histories.append(phase_history)
        current_epoch += epochs

    # Training phase 3: Head tuning
    print("\n=== Phase 3: Final Tuning ===")
    base_model.trainable = False
    for layer in model.layers:
        if "dense" in layer.name or "attention" in layer.name:
            layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-7),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    final_history = model.fit(
        train_gen,
        initial_epoch=current_epoch,
        epochs=current_epoch + final_tune_epochs,
        validation_data=val_gen,
        class_weight=class_weights_dict,
        callbacks=full_callbacks,
        verbose=1
    )
    all_histories.append(final_history)

    # Combine all histories
    combined_history = defaultdict(list)
    for history in all_histories:
        for key, values in history.history.items():
            combined_history[key].extend(values)

    return model, combined_history


# ====================== VISUALIZATION & EVALUATION ======================
def visualize_conv_kernels(model, layer_name, layer_index, max_filters=32, output_dir='conv_kernels'):
    """Visualize convolutional filters for model interpretation"""
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
    """Visualize activation maps from convolutional layers"""
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


def plot_combined_history(history, output_dir='training_history'):
    """
    Visualize training history with correct epoch numbering.
    """
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(1, len(acc) + 1)

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

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "training_history.png"))
    plt.close()


def evaluate_model(model, test_gen, class_names, output_dir='results'):
    """Comprehensive model evaluation with metrics and visualizations"""
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
    """End-to-end execution pipeline"""

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
        output_dir='augmentedImages'  # Save to dedicated folder
    )

    # ====================== MODEL SETUP ======================
    # Construct model architecture
    model = build_resnet_model()  # Custom ResNet-based architecture

    # ====================== TRAINING PHASES ======================
    # Execute multi-phase training process
    trained_model, combined_history = train_model(
        model=model,
        train_gen=train_gen,
        val_gen=val_gen,
        initial_epochs=30,  # Feature extraction phase
        fine_tune_epochs=50,  # Fine-tuning phase
        final_tune_epochs=20  # Final optimization
    )

    # ====================== MODEL MANAGEMENT ======================
    # Model loading with custom objects
    custom_objects = {
        "resnet_preprocess": resnet_preprocess
    }

    if os.path.exists(MODEL_PATH):
        print(f"\nLoading best model from {MODEL_PATH}")
        try:
            best_model = tf.keras.models.load_model(
                MODEL_PATH,
                custom_objects=custom_objects,
                safe_mode=False  # Disable safe mode check
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to newly trained model")
            best_model = trained_model
    else:
        print("\nNo saved model found. Using final trained model.")
        best_model = trained_model
        best_model.save(MODEL_PATH, save_format='keras')

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
        combined_history,
        output_dir='results'  # Save visualization to results/
    )

    # ====================== FINAL EVALUATION ======================
    # Comprehensive model performance assessment
    max_acc = evaluate_model(
        model=best_model,
        test_gen=test_gen,  # Use held-out test set
        class_names=CLASS_NAMES,
        output_dir = 'results'
    )
    print(f"\nMaximum Class Accuracy: {max_acc:.2%}")  # Display peak performance


if __name__ == "__main__":
    main()