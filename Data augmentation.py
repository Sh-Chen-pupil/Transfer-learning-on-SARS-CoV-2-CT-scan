import os
print(os.listdir('./data'))                                
print(os.listdir('./data/train'))                           
print(os.listdir('./data/validation'))                     
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D, Activation
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation


IMAGE_SIZE = 260  # Set the image size


# Training data generator (data augmentation is required)
train_generator = ImageDataGenerator(
    rescale=1./255,             # Rescale pixel values to the range of 0-1
    rotation_range=360,         # Image rotation
    width_shift_range=0.2,      # Horizontal shifting
    height_shift_range=0.2,     # Vertical shifting
    zoom_range=0.2,             # Image zooming
    horizontal_flip=True,       # Random horizontal flipping
    vertical_flip=True
).flow_from_directory(
    './data/train',             # Set the directory for reading data
    target_size=(IMAGE_SIZE, IMAGE_SIZE),  # Resize images to a uniform size
    batch_size=64,               # Set the batch size to 64
    class_mode='categorical',   # Declare it as a categorical classification problem
    shuffle=True                # Shuffle the data before each epoch
)


# Validation data generator (no data augmentation required)
validation_generator = ImageDataGenerator(
    rescale=1./255              # Rescale pixel values to the range of 0-1
).flow_from_directory(
    './data/validation',        # Set the directory for reading data
    target_size=(IMAGE_SIZE, IMAGE_SIZE),  # Resize images to a uniform size
    batch_size=64,               # Set the batch size to 64
    class_mode='categorical',   # Declare it as a categorical classification problem
    shuffle=False               # Do not shuffle the order before each epoch (useful for subsequent predictions)
)


X, Y = next(train_generator)      # Get a batch of training data
fig, ax = plt.subplots(2, 6)      # Create a canvas and split it into 2 rows and 6 columns
fig.set_figheight(20)            # Set the height of the canvas
fig.set_figwidth(50)             # Set the width of the canvas
ax = ax.flatten()                # Flatten the canvas
for i in range(12):              # Display images one by one
    ax[i].imshow(X[i])           # Display the training image
    ax[i].set_title(Y[i, 1], fontsize=25)  # Use the Y value as the title




