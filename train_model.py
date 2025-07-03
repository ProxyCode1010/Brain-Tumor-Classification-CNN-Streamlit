import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

IMG_SIZE = (150, 150)
BATCH_SIZE = 16

# Step 1: Setup data generator with validation split
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Step 2: Load training data (80%)
train_data = train_datagen.flow_from_directory(
    'Training',  # ✅ folder name changed from 'dataset' to 'Training'
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',   # ✅ changed from 'binary'
    subset='training'
)

# Step 3: Load validation data (20%)
val_data = train_datagen.flow_from_directory(
    'Training',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',   # ✅ changed from 'binary'
    subset='validation'
)

# Step 4: Create CNN model for multi-class
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')  # ✅ 4 classes, use softmax
])

# Step 5: Compile the model with categorical loss
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 6: Train the model
model.fit(train_data, validation_data=val_data, epochs=10)  #epochs parameter in model training is one of the most important hyperparameters in deep learning. More epochs more accuracy upto particular range

# Step 7: Save the model
model.save("model/brain_tumor_model.h5")