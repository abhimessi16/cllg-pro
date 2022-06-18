import os
import shutil
from tokenize import String
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.layers import *
from keras.models import *
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import cv2
import numpy as np

start=time.time()

datset_dir="./Dataset/AdienceBenchmarkGenderAndAgeClassification/"

print('starting model creation')

inputs = tf.keras.layers.Input(shape=(64,64,1))
conv1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3),activation='relu')(inputs)
conv2 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3),activation='relu')(conv1)
pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3),activation='relu')(pool1)
pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
x = tf.keras.layers.Dropout(0.25)(pool2)
flat = tf.keras.layers.Flatten()(x)

print('basic model done!!')

dropout = tf.keras.layers.Dropout(0.5)
age_model = tf.keras.layers.Dense(128, activation='relu')(flat)
age_model = dropout(age_model)
age_model = tf.keras.layers.Dense(64, activation='relu')(age_model)
age_model = dropout(age_model)
age_model = tf.keras.layers.Dense(32, activation='relu')(age_model)
age_model = dropout(age_model)
age_model = tf.keras.layers.Dense(1, activation='relu')(age_model)

print('age model done!!!')

dropout = tf.keras.layers.Dropout(0.5)
gender_model = tf.keras.layers.Dense(128, activation='relu')(flat)
gender_model = dropout(gender_model)
gender_model = tf.keras.layers.Dense(64, activation='relu')(gender_model)
gender_model = dropout(gender_model)
gender_model = tf.keras.layers.Dense(32, activation='relu')(gender_model)
gender_model = dropout(gender_model)
gender_model = tf.keras.layers.Dense(16, activation='relu')(gender_model)
gender_model = dropout(gender_model)
gender_model = tf.keras.layers.Dense(8, activation='relu')(gender_model)
gender_model = dropout(gender_model)
gender_model = tf.keras.layers.Dense(1, activation='sigmoid')(gender_model)

print('gender model done')

model = tf.keras.models.Model(inputs=inputs, outputs=[age_model,gender_model])

print('mapping output done')

model.compile(optimizer = 'adam', loss =['mse','binary_crossentropy'],metrics=['accuracy'])

print('model compiled')

# datadir='./data_dir/'
# validationdir='./Validation/'

# train_datagen=ImageDataGenerator(rescale=1/255)
# validate_datagen=ImageDataGenerator(rescale=1/255)

# train_len,test_len=19370,11649

# training_generator=train_datagen.flow_from_directory(datadir, target_size=(300,300), batch_size=20, class_mode='binary')
# validation_generator=validate_datagen.flow_from_directory(validationdir, target_size=(300,300), batch_size=20, class_mode='binary')


# h = model.fit(x_train,[y_train[:,0],y_train[:,1]],validation_data=(x_test,[y_test[:,0],y_test[:,1]]),epochs = 25, batch_size=128,shuffle = True)
# history=gender_model.fit(training_generator, steps_per_epoch=(train_len//80), epochs=25, validation_data=validation_generator, validation_steps=(test_len//80), verbose=1)

# gender_model.save('gender_model.h5')

# end=time.time()

# print('done after {}'.format(end-start))

# """
# path='./testing/091270.jpg'

# image1=image.load_img(path, target_size=(300,300))
# image2=image.img_to_array(image1)
# image2=np.expand_dims(image2, axis=0)

# img_shows=np.vstack([image2])

# prediction=model.predict(img_shows, batch_size=10)

# print(prediction)

# """