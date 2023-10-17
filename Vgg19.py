from keras.applications import vgg19
base_model3 = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
x = base_model3.output  # Get the output of the base model.
x = GlobalAveragePooling2D()(x)  # Global average pooling.
x = BatchNormalization()(x)  # Batch normalization layer.
x = Dropout(0.4)(x)  # Dropout to prevent overfitting.
x = Dense(256, activation='relu')(x)  # The first fully connected layer.
x = BatchNormalization()(x)  # Another batch normalization operation.
x = Dropout(0.3)(x)  # Another dropout operation.
predictions3 = Dense(2, activation='softmax')(x)  # The second fully connected layer for output.
model3 = Model(inputs=base_model3.input, outputs=predictions3)  # Build the model, specifying the input and output.
for layer in base_model3.layers:
    layer.trainable = False  # Set the layers in the base model to be non-trainable.


model3.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])
model3.fit(train_generator,                                                  
          epochs=20,                                                       
          validation_data=validation_generator,workers = 12)      
