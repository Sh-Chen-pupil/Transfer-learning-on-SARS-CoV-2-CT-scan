from tensorflow.keras.applications.resnet50 import ResNet50


base_model2 = ResNet50(include_top=False, weights='imagenet')
x = base_model2.output                      # Get the output of the base model.
x = GlobalAveragePooling2D()(x)             # Global average pooling.
x = BatchNormalization()(x)                 # Batch normalization layer.
x = Dropout(0.4)(x)                         # Dropout to prevent overfitting.
x = Dense(256, activation='relu')(x)        # The first fully connected layer.
x = BatchNormalization()(x)                 # Another batch normalization operation.
x = Dropout(0.3)(x)                         # Another dropout operation.
predictions2 = Dense(2, activation='softmax')(x)  # The second fully connected layer for output.
model2 = Model(inputs=base_model2.input, outputs=predictions2)  # Build the model, specifying the input and output.
for layer in base_model2.layers:
    layer.trainable = False                 # Set the layers in the base model to be non-trainable.


model2.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])
model2.fit(train_generator,                                                  
          epochs=20,                                                        
          validation_data=validation_generator,workers = 12)      
