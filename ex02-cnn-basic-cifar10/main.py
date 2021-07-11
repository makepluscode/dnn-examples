IMAGE_SIZE = 32

from os import name
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Dropout, Flatten, Activation, MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler

# KerasTensor(type_spec=TensorSpec(shape=(None, 32, 32, 3),
# dtype=tf.float32, name='input_1'), name='input_1', description="created by layer 'input_1'")
input_tensor = Input(shape=(IMAGE_SIZE, IMAGE_SIZE,3))

print(input_tensor)

x = Conv2D(filters=16, kernel_size=(5, 5), padding='valid', activation='relu')(input_tensor)
x = Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(x)
x = MaxPool2D(pool_size=(2,2))(x)

x = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(x)
x = Conv2D(filters=32, kernel_size=3, padding='same')(x)
x = Activation('relu')(x)
x = MaxPool2D(pool_size=(2,2))(x)

x = Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu')(x)
x = Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu')(x)
x = MaxPool2D(pool_size=(2,2))(x)

x = Flatten(name='flatten')(x)
x = Dropout(rate=0.5)(x)
x = Dense(300, activation='relu', name='fc1')(x)
x = Dropout(rate=0.3)(x)
output = Dense(10, activation='softmax', name='output')(x)

model = Model(inputs=input_tensor, outputs=output)

model.summary()

model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

### Model Visualization
from tensorflow.keras.utils import plot_model

plot_model(model, to_file='model.png')
plot_model(model, to_file='model_shapes.png', show_shapes=True)

### Trainning Datasets
import numpy as np
import pandas as pd

import os

from tensorflow.keras.datasets import cifar10

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
print (train_images[0, :, :, :], train_labels[0, :])

NAMES = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
print(train_labels[0:10])

import matplotlib.pyplot as plt
import cv2

def show_images(images, labels, ncols=8):
    figure, axs = plt.subplots(figsize=(22, 6), nrows=1, ncols=ncols)
    for i in range(ncols):
        axs[i].imshow(images[i])
        label = labels[i].squeeze()
        axs[i].set_title(NAMES[int(label)])

show_images(train_images[:8], train_labels[:8], ncols=8)
show_images(train_images[8:16], train_labels[8:16], ncols=8)


#history = model.fit(x=train_images,
