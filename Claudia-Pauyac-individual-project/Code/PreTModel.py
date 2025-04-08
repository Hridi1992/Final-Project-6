import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import f1_score

# Define paths
train_dir = r'C:\Users\User\Documents\OsuSpring2025\DeepLearning\FProject\.venv\train'  # Contains class subfolders
test_dir = r'C:\Users\User\Documents\OsuSpring2025\DeepLearning\FProject\.venv\test'

# Image parameters
IMG_SIZE = 48  # Original FER2013 size
TARGET_SIZE = 224  # ResNet50 input size
BATCH_SIZE = 32
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def create_data_generators():
    """Create and configure data generators for training, validation, and testing"""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        classes=CLASS_NAMES,
        subset='training'
    )

    val_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        classes=CLASS_NAMES,
        shuffle=False,
        subset='validation'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        classes=CLASS_NAMES,
        shuffle=False
    )

    return train_generator, val_generator, test_generator

# 2. Build model with integrated preprocessing
def build_resnet_model():
    # Input and augmentation
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))

    # Data augmentation
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.15)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.RandomContrast(0.2)(x)

    # ResNet50 preprocessing
    x = layers.Resizing(TARGET_SIZE, TARGET_SIZE)(x)
    x = layers.Concatenate()([x, x, x])  # Grayscale to RGB
    x = layers.Lambda(preprocess_input)(x)  # Official ResNet50 preprocessing

    # Base model
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(TARGET_SIZE, TARGET_SIZE, 3)
    )
    x = base_model(x)

    # New: Additional Conv Layer
    x = layers.Conv2D(512, (1, 1), activation='relu')(x)

    # Classifier
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    # New: Additional Dense Layer
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.L2(0.01))(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(7, activation='softmax')(x)

    return models.Model(inputs,outputs)


# --------------------------------------------------
# 4. Initial Training Phase (Frozen Base Model)
# --------------------------------------------------
# --------------------------------------------------
# 5. Fine-Tuning Phase (Partial Unfreezing)
# --------------------------------------------------
def train_model(model, train_gen, val_gen, initial_epochs=15, fine_tune_epochs=25):
    """Handle both initial training and fine-tuning phases"""
    # Initial training with frozen base
    print("\n=== Initial Training ===")
    model.get_layer("resnet50").trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    initial_history = model.fit(
        train_gen,
        epochs=initial_epochs,
        validation_data=val_gen,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint('initial_best.h5', save_best_only=True)
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
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    fine_tune_history = model.fit(
        train_gen,
        epochs=initial_epochs + fine_tune_epochs,
        initial_epoch=initial_history.epoch[-1] + 1,
        validation_data=val_gen,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=3),
            tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
        ]
    )

    return model


def evaluate_model(model, test_gen):
    """Evaluate model performance with accuracy, F1 score, and confusion matrix"""
    print("\n=== Final Evaluation ===")

    # Basic evaluation
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"Test Accuracy: {test_acc:.2%}")

    # Generate predictions
    y_pred_probs = model.predict(test_gen)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_gen.classes

    # F1 Score
    f1 = f1_score(y_true, y_pred, average='macro')
    print(f"F1 Macro Score: {f1:.2%}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    # Visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
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

    # Training pipeline
    trained_model = train_model(model, train_gen, val_gen)

    # Load best model and evaluate
    best_model = models.load_model('best_model.h5')
    evaluate_model(best_model, test_gen)


if __name__ == "__main__":
    main()