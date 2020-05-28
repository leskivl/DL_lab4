import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt


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
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=2000 // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=1000 // batch_size
)
history_dict = history.history
print(history_dict.keys())
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

from PIL import Image, ImageOps
amount=999
current=0
for i in range(999):
    im = Image.open(train_cats_dir + '/cat.'+ str(i) + '.jpg')
    im_flip = ImageOps.flip(im)
    amount+=1
    im_flip.save(train_cats_dir + '/cat' + str(amount) + '.jpg', quality=95)
    im_mirror = ImageOps.mirror(im)
    amount+=1
    im_mirror.save(train_cats_dir + '/cat' + str(amount) + '.jpg', quality=95)
