import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define constants
IMG_HEIGHT = 48
IMG_WIDTH = 48
NUM_SAMPLES = 100
BATCH_SIZE = 32
NUM_CLASSES = 7  # Emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
EPOCHS = 20  # Reduced epochs due to small dataset
EMOTIONS = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
SPLIT_RATIOS = {'Training': 0.7, 'PublicTest': 0.15, 'PrivateTest': 0.15}

# Generate synthetic pixel data for an image
def generate_synthetic_pixels(emotion_id):
    image = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    if emotion_id == 0:  # Angry: Diagonal lines
        for i in range(0, IMG_HEIGHT, 4):
            image[i:i+2, :] = 200
    elif emotion_id == 1:  # Disgust: Checkerboard
        image[::2, ::2] = 150
        image[1::2, 1::2] = 150
    elif emotion_id == 2:  # Fear: Random noise
        image = np.random.randint(100, 200, (IMG_HEIGHT, IMG_WIDTH))
    elif emotion_id == 3:  # Happy: Smiley face pattern
        image[15:33, 15:33] = 200
        image[20:22, 20:28] = 50
        image[28:30, 22:26] = 50
    elif emotion_id == 4:  # Sad: Downward curve
        for i in range(IMG_HEIGHT):
            image[i, 10:38] = 100 if i > 24 else 50
    elif emotion_id == 5:  # Surprise: Circles
        image[10:38, 10:38] = 180
        image[20:28, 20:28] = 80
    elif emotion_id == 6:  # Neutral: Uniform gray
        image.fill(128)
    noise = np.random.randint(-20, 20, (IMG_HEIGHT, IMG_WIDTH))
    image = np.clip(image + noise, 0, 255)
    pixels = image.flatten()
    pixel_str = ' '.join(map(str, pixels))
    return pixel_str

# Generate synthetic dataset
def create_synthetic_dataset():
    data = {'emotion': [], 'pixels': [], 'usage': []}
    num_training = int(NUM_SAMPLES * SPLIT_RATIOS['Training'])
    num_public_test = int(NUM_SAMPLES * SPLIT_RATIOS['PublicTest'])
    num_private_test = NUM_SAMPLES - num_training - num_public_test
    splits = (
        ['Training'] * num_training +
        ['PublicTest'] * num_public_test +
        ['PrivateTest'] * num_private_test
    )
    for i in range(NUM_SAMPLES):
        emotion_id = i % NUM_CLASSES
        pixel_str = generate_synthetic_pixels(emotion_id)
        data['emotion'].append(emotion_id)
        data['pixels'].append(pixel_str)
        data['usage'].append(splits[i])
    df = pd.DataFrame(data)
    output_file = 'fer2013_synthetic.csv'
    df.to_csv(output_file, index=False)
    print(f"Synthetic dataset saved as '{output_file}'")
    return df

# Load and preprocess dataset
def load_dataset():
    df = create_synthetic_dataset() if not os.path.exists('fer2013_synthetic.csv') else pd.read_csv('fer2013_synthetic.csv')
    pixels = df['pixels'].tolist()
    emotions = df['emotion'].tolist()
    images = []
    for pixel_sequence in pixels:
        pixel_array = [int(pixel) for pixel in pixel_sequence.split()]
        image = np.array(pixel_array).reshape(IMG_HEIGHT, IMG_WIDTH, 1)
        images.append(image)
    images = np.array(images, dtype='float32') / 255.0
    emotions = tf.keras.utils.to_categorical(emotions, NUM_CLASSES)
    return images, emotions

# Build CNN model
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Data augmentation
def create_data_generator():
    return ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

# Train and save model
def train_and_save_model():
    images, emotions = load_dataset()
    X_train, X_val, y_train, y_val = train_test_split(images, emotions, test_size=0.2, random_state=42)
    datagen = create_data_generator()
    datagen.fit(X_train)
    model = build_model()
    model.summary()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint('emotion_model_checkpoint.h5', save_best_only=True, monitor='val_accuracy'),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)
    ]
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        callbacks=callbacks
    )
    model.save('emotion_detection_model.h5')
    print("Model saved as 'emotion_detection_model.h5'")
    return history

if __name__ == '__main__':
    history = train_and_save_model()
    print("Training completed. Accuracy on validation set:", history.history['val_accuracy'][-1])