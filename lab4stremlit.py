import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pydot_ng as pydot
import streamlit as st
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import os
from matplotlib.pyplot import imshow
from PIL import Image, ImageOps


st.title('Convolutional Neural Network')
st.header('Dataset: cat_or_dog')


train_dir = ('./train')
validation_dir = ('./validation')
train_cats_dir = ('./train/cats')
train_dogs_dir = ('./train/dogs')
validation_cats_dir = ('./validation/cats')
validation_dogs_dir = ('./validation/dogs')

batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150

train_image_generator = ImageDataGenerator(rescale=1./255)
validation_image_generator = ImageDataGenerator(rescale=1./255)

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')
val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')


st.subheader('Inspecting dataset')
if st.checkbox('Show random image from the train set'):
    num = np.random.randint(0, 1000)
    image = Image.open(train_cats_dir + '/cat.'+ str(num) + '.jpg','r')
    st.image(image)

st.subheader('Set some hyperparameters')
batch_size = st.selectbox('Select batch size', [16, 32, 64, 128, 256])
epochs=st.selectbox('Select number of epochs', [1, 3, 10, 25, 50])
loss_function = st.selectbox('Loss function', ['binary_crossentropy','categorical_crossentropy', 'mean_squared_error', 'mean_absolute_error'])
optimizer = st.selectbox('Optimizer', ['Adam', 'RMSprop', 'SGD'])

st.subheader('Building your CNN')

if st.checkbox('Fit model'):
    model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')])
    model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])

    history = model.fit_generator(
        train_data_gen,
        steps_per_epoch=2000 // batch_size,
        epochs=epochs,
        validation_data=val_data_gen,
        validation_steps=1000 // batch_size
    )

    st.write("loss:")
    st.write(history.history['loss'])



st.subheader('Visualizing results')

if st.checkbox('Loss visualization:'):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    st.pyplot()

if st.checkbox('Show random prediction results:'):
    num = np.random.randint(0, 1000)
    pred = model.predict(val_data_gen)
    image = Image.open(train_cats_dir + '/cat.'+ str(num) + '.jpg','r')
    st.image(image)
    if pred[num] > 5:
      st.write("Dog")
    else:
      st.write("Cat")
