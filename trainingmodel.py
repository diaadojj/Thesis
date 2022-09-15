import numpy as np
#import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation,Flatten,Conv2D, MaxPooling2D
import tensorflow.keras as keras
from tensorflow.keras import layers
from keras.preprocessing.image import ImageDataGenerator

def create_model():
    model = Sequential([
  # Note the input shape is the size of the image 110x220 with 3 bytes color
  # This is the first convolution
    Conv2D(32, (3,3), activation='relu', input_shape=(110, 220, 3)
    ),
    MaxPooling2D(2, 2),
  # The second convolution
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
  # The third convolution
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
  # The fourth convolution
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
  # Flatten the results to feed into a DNN
    Flatten(),
    Dropout(0.5),
  # 512 neuron hidden layer
    Dense(512, activation='relu'),
    Dense(17, activation='softmax')
    ])
    model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

batch_size = 100

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
       )

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        'C:/Users/SAHIN/Desktop/New training/Dataset1',  # this is the target directory
        target_size=(110, 220),  # all images will be resized to 110x220
        batch_size=batch_size,
        class_mode='categorical')  # 

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        'C:/Users/SAHIN/Desktop/New training/test',
        target_size=(110, 220),
        batch_size=batch_size,
        class_mode='categorical', shuffle=False)

callbacks = tf.keras.callbacks.ModelCheckpoint(filepath='C:/Users/SAHIN/Desktop/New training/best_model.h5',monitor='val_loss', mode
='min', save_best_only=True, verbose=1)

myModel = create_model()
myModel.fit_generator(
        train_generator,
        steps_per_epoch=  train_generator.samples   // batch_size,
        epochs=100,
        callbacks=callbacks,
        
        validation_data=validation_generator,
        validation_steps= validation_generator.samples // batch_size, verbose = 1)
#model.save_weights('/content/drive/MyDrive/Colabs/second_try2.h5')  # always save your weights after training or during training
myModel.save('/myModel.h5')