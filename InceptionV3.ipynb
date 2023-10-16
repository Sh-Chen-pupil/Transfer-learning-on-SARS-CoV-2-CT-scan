from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D, Activation
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation


base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False)  # Base model Inception V3; Set transfer model to exclude fully connected layers.
x = base_model.output  # Get the output of the base model.
x = GlobalAveragePooling2D()(x)  # Global average pooling.
x = BatchNormalization()(x)  # Batch normalization layer.
x = Dropout(0.4)(x)  # Dropout to prevent overfitting.
x = Dense(300, activation='relu')(x)  # The first fully connected layer.
x = BatchNormalization()(x)  # Another batch normalization operation.
x = Dropout(0.3)(x)  # Another dropout operation.
predictions = Dense(2, activation='softmax')(x)  # The second fully connected layer for output.
model = Model(inputs=base_model.input, outputs=predictions)  # Build the model, specifying the input and output.
for layer in base_model.layers:
    layer.trainable = False  # Set the layers in the base model to be non-trainable.


nb_layers = len(base_model.layers)
print(base_model.layers[nb_layers - 2].name)
print(base_model.layers[nb_layers - 1].name)


model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])
model.fit(
    train_generator,               # Specify the training dataset
    epochs=20,                    # Set the number of epochs
    validation_data=validation_generator, workers=12  # Set the validation dataset and enable multi-threaded computation
)
