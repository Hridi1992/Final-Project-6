import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D, Concatenate, BatchNormalization, Conv2D
from tensorflow.keras.layers import Resizing, Lambda
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.saving import register_keras_serializable
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.utils import class_weight  # For class weights
import os

# Define paths (adjust as needed)
TRAIN_DIR = r'/home/ubuntu/Project/archive/new/train'
TEST_DIR = r'/home/ubuntu/Project/archive/new/test'

# Image parameters
IMG_SIZE = 48  # Original FER2013 size
TARGET_SIZE = 224  # ResNet50 input size
BATCH_SIZE = 32
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
NUM_CLASSES = len(CLASS_NAMES)

@register_keras_serializable()
def resnet_preprocess(x):
    """Custom preprocessing for ResNet50"""
    return preprocess_input(x)

# Custom weighted categorical crossentropy loss
@register_keras_serializable()
class WeightedCategoricalCrossentropy(tf.keras.losses.Loss):
    def __init__(self, class_weights, label_smoothing=0.0, name='weighted_categorical_crossentropy'):
        super().__init__(name=name)
        self.class_weights = tf.constant(list(class_weights.values()), dtype=tf.float32)
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            num_classes = tf.cast(tf.shape(y_true)[-1], y_pred.dtype)
            y_true = y_true * (1.0 - self.label_smoothing) + (self.label_smoothing / num_classes)

        # Compute crossentropy
        loss = -y_true * tf.math.log(y_pred)
        loss = tf.reduce_sum(loss, axis=-1)

        # Apply class weights
        weights = tf.reduce_sum(self.class_weights * y_true, axis=-1)
        loss = loss * weights

        return tf.reduce_mean(loss)

    def get_config(self):
        config = super().get_config()
        config.update({
            'class_weights': {i: w for i, w in enumerate(self.class_weights.numpy())},
            'label_smoothing': self.label_smoothing
        })
        return config

# MixUp generator for augmentation
def mixup_generator(gen, alpha=0.2):
    """Yield MixUp-augmented batches from a Keras generator."""
    while True:
        x1, y1 = next(gen)
        x2, y2 = next(gen)
        if x1.shape[0] != x2.shape[0]:
            continue
        lam = np.random.beta(alpha, alpha)
        x = lam * x1 + (1 - lam) * x2
        y = lam * y1 + (1 - lam) * y2
        yield x, y

class BaseModel:
    def __init__(self):
        self.model = None
        self.trained = False
        self.history = None

    def model_builder(self, in_shape, out_shape, optim, loss, **kwargs):
        raise NotImplementedError("model_builder is not implemented")

    def train(self):
        raise NotImplementedError("train is not implemented")

    def evaluate(self, test_gen, save_evaluation_to=None):
        if self.trained:
            # Evaluate on the entire test set
            loss, acc = self.model.evaluate(test_gen)
            print(f"Test Accuracy: {acc:.2%}")

            # Predict on the entire test set
            test_gen.reset()  # Reset generator to start from the beginning
            yhat_test = np.argmax(self.model.predict(test_gen), axis=1)
            ytest_ = test_gen.classes  # Get true labels for the entire test set

            f1 = f1_score(ytest_, yhat_test, average='macro')
            print(f"F1 Macro Score: {f1:.2%}")
            print(classification_report(ytest_, yhat_test, target_names=CLASS_NAMES, zero_division=0))

            if save_evaluation_to is not None:
                cm = confusion_matrix(ytest_, yhat_test)
                plt.figure(figsize=(7, 7))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
                plt.title('Confusion Matrix')
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.savefig(save_evaluation_to)
                plt.close()
        else:
            raise ValueError("Model is not trained yet, call train first")

    def predict(self, X, classes=True):
        if self.trained:
            return np.argmax(self.model.predict(X), axis=1) if classes else self.model.predict(X)
        else:
            raise ValueError("Model is not trained yet, call train first")

    def save_model(self, path):
        if self.trained:
            self.model.save(path)
        else:
            raise ValueError("Model is not trained yet, call train first")

    def save_training_history(self, path):
        if self.trained:
            sns.set()
            fig = plt.figure(0, (12, 4))

            ax = plt.subplot(1, 2, 1)
            sns.lineplot(x=self.history.epoch, y=self.history.history['accuracy'], label='train')
            try:
                sns.lineplot(x=self.history.epoch, y=self.history.history['val_accuracy'], label='valid')
            except KeyError:
                pass
            plt.title('Accuracy')
            plt.tight_layout()

            ax = plt.subplot(1, 2, 2)
            sns.lineplot(x=self.history.epoch, y=self.history.history['loss'], label='train')
            try:
                sns.lineplot(x=self.history.epoch, y=self.history.history['val_loss'], label='valid')
            except KeyError:
                pass
            plt.title('Loss')
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
        else:
            raise ValueError("Model is not trained yet, call train first")

    def __repr__(self):
        return f"model: {self.__class__.__name__}, trained: {self.trained}"

class ResNet50Model(BaseModel):
    def model_builder(self, in_shape, out_shape):
        # Input layer for grayscale images
        inputs = Input(shape=in_shape, name="input_resnet50")

        # Enhanced data augmentation
        x = tf.keras.layers.RandomFlip("horizontal")(inputs)
        x = tf.keras.layers.RandomRotation(0.3)(x)
        x = tf.keras.layers.RandomZoom(0.2)(x)
        x = tf.keras.layers.RandomContrast(0.2)(x)
        x = tf.keras.layers.RandomTranslation(0.2, 0.2)(x)

        # Preprocessing: Resize and convert grayscale to RGB
        x = Resizing(TARGET_SIZE, TARGET_SIZE)(x)
        x = Concatenate()([x, x, x])  # Convert grayscale to 3-channel
        x = Lambda(resnet_preprocess)(x)

        # Load pretrained ResNet50
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(TARGET_SIZE, TARGET_SIZE, 3),
            name='resnet50'
        )
        x = base_model(x)

        # Add a convolutional layer to better extract features
        x = Conv2D(512, (3, 3), padding='same', activation='relu',
                   kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = BatchNormalization()(x)

        # Additional layers with reduced complexity
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        # Output layer
        outputs = Dense(out_shape, activation='softmax', name='out_layer')(x)

        self.model = Model(inputs=inputs, outputs=outputs, name="ResNet50_Emotion")

    def train(self, train_gen, val_gen, batch_size=32, initial_epochs=10, fine_tune_epochs=10,
              optim=optimizers.Adam(1e-4), callbacks=[]):
        # Compute class weights for imbalanced data with smoothing
        cls = train_gen.classes
        cw = class_weight.compute_class_weight('balanced', classes=np.unique(cls), y=cls)
        # Smooth class weights to prevent overcompensation
        smoothed_cw = np.clip(cw, 0.5, 2.0)  # Limit weights between 0.5 and 2.0
        class_weights = dict(enumerate(smoothed_cw))

        # Build model
        self.model_builder(in_shape=(IMG_SIZE, IMG_SIZE, 1), out_shape=NUM_CLASSES)

        # Initial training with frozen base
        print("\n=== Initial Training ===")
        self.model.get_layer("resnet50").trainable = False

        # Use custom weighted loss
        weighted_loss = WeightedCategoricalCrossentropy(class_weights, label_smoothing=0.1)
        self.model.compile(
            optimizer=optim,
            loss=weighted_loss,
            metrics=['accuracy']
        )

        # Use MixUp for initial training
        steps = len(train_gen)
        mixup_gen = mixup_generator(train_gen, alpha=0.2)
        initial_history = self.model.fit(
            mixup_gen,
            steps_per_epoch=steps,
            epochs=initial_epochs,
            validation_data=val_gen,
            callbacks=callbacks + [
                tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                tf.keras.callbacks.ModelCheckpoint('initial_best.keras', save_best_only=True)
            ]
        )

        # Fine-tuning
        print("\n=== Fine-Tuning ===")
        base_model = self.model.get_layer("resnet50")
        base_model.trainable = True

        # Freeze first 140 layers (approx. 80% of 175 layers)
        for layer in base_model.layers[:140]:
            layer.trainable = False

        # Recompile with the same weighted loss
        self.model.compile(
            optimizer=optimizers.Adam(
                learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=1e-4,
                    decay_steps=3000,
                    decay_rate=0.9
                ),
                clipnorm=1.0
            ),
            loss=weighted_loss,
            metrics=['accuracy']
        )

        self.history = self.model.fit(
            train_gen,
            epochs=initial_epochs + fine_tune_epochs,
            initial_epoch=initial_history.epoch[-1] + 1,
            validation_data=val_gen,
            callbacks=callbacks + [
                tf.keras.callbacks.EarlyStopping(patience=5),
                tf.keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True)
            ]
        )

        self.trained = True

def create_data_generators():
    """Create data generators for training, validation, and testing"""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.2,
        zoom_range=0.2,
        brightness_range=(0.8, 1.2),
        horizontal_flip=True,
        validation_split=0.2
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_NAMES,
        subset='training'
    )

    val_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_NAMES,
        shuffle=False,
        subset='validation'
    )

    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_NAMES,
        shuffle=False
    )

    return train_generator, val_generator, test_generator

def print_class_distribution(data_dir):
    """Print number of images per class in a directory"""
    print("\n=== Class Distribution ===")
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(data_dir, class_name)
        if os.path.exists(class_dir):
            num_images = len(os.listdir(class_dir))
            print(f"{class_name.capitalize():<9}: {num_images} images")
        else:
            print(f"{class_name.capitalize():<9}: Directory not found")

def main():
    # Initialize data generators
    train_gen, val_gen, test_gen = create_data_generators()

    # Print class distribution
    print("\n=== Training dataset ===")
    print_class_distribution(TRAIN_DIR)
    print("\n=== Testing dataset ===")
    print_class_distribution(TEST_DIR)

    # Initialize and train model
    model = ResNet50Model()
    model.train(
        train_gen=train_gen,
        val_gen=val_gen,
        batch_size=BATCH_SIZE,
        initial_epochs=10,
        fine_tune_epochs=10,
        callbacks=[]
    )

    # Evaluate model
    model.evaluate(test_gen, save_evaluation_to='confusion_matrix.png')

    # Save model and training history
    model.save_model('resnet50_emotion_model.keras')
    model.save_training_history('training_history.png')

if __name__ == "__main__":
    main()
