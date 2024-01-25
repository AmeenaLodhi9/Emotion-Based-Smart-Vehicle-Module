from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

IMG_HEIGHT = 227
IMG_WIDTH = 227
batch_size = 32
epochs = 1

# Replace these paths with the actual paths to your dataset
import os


# Get the absolute path of the current working directory
current_directory = os.getcwd()

# Use absolute paths
train_data_dir = os.path.join(current_directory, 'dataset_drowsiness', 'Driver Drowsiness Dataset (DDD)', 'train')
validation_data_dir = os.path.join(current_directory, 'dataset_drowsiness', 'Driver Drowsiness Dataset (DDD)', 'test')





# Data augmentation for training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# Rescaling for validation set (no augmentation)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Data generators
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch_size,
    class_mode='binary',  # binary for drowsiness detection
    shuffle=True)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch_size,
    class_mode='binary',  # binary for drowsiness detection
    shuffle=True)

# Model for drowsiness detection
drowsiness_model = Sequential()

drowsiness_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
drowsiness_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
drowsiness_model.add(MaxPooling2D(pool_size=(2, 2)))
drowsiness_model.add(Dropout(0.3))

drowsiness_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
drowsiness_model.add(MaxPooling2D(pool_size=(2, 2)))
drowsiness_model.add(Dropout(0.3))

drowsiness_model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
drowsiness_model.add(MaxPooling2D(pool_size=(2, 2)))
drowsiness_model.add(Dropout(0.3))

drowsiness_model.add(Flatten())
drowsiness_model.add(Dense(512, activation='relu'))
drowsiness_model.add(Dropout(0.2))

drowsiness_model.add(Dense(1, activation='sigmoid'))  # Binary output for drowsiness detection

# Compile the model
drowsiness_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(drowsiness_model.summary())

# Train the model for drowsiness detection
history = drowsiness_model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=len(validation_generator))

# Save the trained drowsiness detection model
drowsiness_model.save('drowsiness_detection_model.h5')
