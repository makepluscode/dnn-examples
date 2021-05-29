
#######################################################################
import tensorflow as tf
import numpy as np
import pandas as pd

import random as python_random
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar10

def set_random_seed(seed_value):
    np.random.seed(seed_value)
    python_random.seed(seed_value)
    tf.random.set_seed(seed_value)

def get_preprocessed_data(images, labels, scaling=True):
    if scaling:
        image = np.array(images/255.0, dtype=np.float32)
    else:
        image = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    return images, labels

def get_preprocessed_ohe(images, labels):
    images, label = get_preprocessed_data(images, labels, scaling=False)
    oh_labels = to_categorical(labels)
    return images, oh_labels

def get_train_vaild_test_set(train_images, train_labels, test_images, test_labels, valid_size=0.15, random_state=2021):
    train_images, train_oh_labels = get_preprocessed_ohe(train_images, train_labels)
    test_images, test_oh_labels = get_preprocessed_ohe(test_images, test_labels)

    tr_images, val_images, tr_oh_lables, val_oh_labels = train_test_split(train_images, train_oh_labels, test_size=valid_size, random_state=random_state)

    return (tr_images, tr_oh_lables), (val_images, val_oh_labels), (test_images, test_oh_labels)

set_random_seed(2021)
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

print(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)

(tr_images, tr_oh_labels), (val_images, val_oh_labels), (test_images, test_oh_labels) = get_train_vaild_test_set(train_images, train_labels, test_images, test_labels, valid_size=0.15, random_state=2021)

print(tr_images.shape, tr_oh_labels.shape,  val_images.shape, val_oh_labels.shape, test_labels.shape, test_images.shape, test_oh_labels.shape)

#######################################################################
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BATCH_SIZE = 64

train_generator = ImageDataGenerator(
    horizontal_flip=True,
    rescale=1/255.0
)

vaild_generator = ImageDataGenerator(
    rescale=1/255.0
)

flow_tr_gen = train_generator.flow(tr_images, tr_oh_labels, batch_size=BATCH_SIZE, shuffle=True)
flow_val_gen = vaild_generator.flow(val_images, val_oh_labels, batch_size=BATCH_SIZE, shuffle=False)

#######################################################################
from tensorflow.keras.applications import VGG16, ResNet50, ResNet50V2, Xception
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Dropout, Flatten, Activation, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam, RMSprop

IMAGE_SIZE = 32

def create_model(verbose=False):
    input_tensor = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))    
    base_model = VGG16(
        input_tensor=input_tensor,
        include_top=False,
        weights='imagenet'
    )
    bm_output = base_model.output

    x = GlobalAveragePooling2D()(bm_output)
    x = Dense(50, activation='relu', name='fc1')(x)
    output = Dense(10, activation='softmax', name='output')(x)

    model = Model(inputs=base_model.input, outputs=output)
    if verbose:
        model.summary()

    return model

vgg_model = create_model(verbose=True)
vgg_model.compile(
    optimizer=Adam(lr=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

tr_data_len = tr_images.shape[0]
val_data_len = val_images.shape[0]

history = vgg_model.fit(
    flow_tr_gen,
    epochs=10,
    steps_per_epoch=int(np.ceil(tr_data_len/BATCH_SIZE)),
    validation_data=flow_val_gen,
    validation_steps=int(np.ceil(val_data_len/BATCH_SIZE)),
)

vgg_model.save("my_model")

test_generator = ImageDataGenerator(rescale=1/255.0)
flow_test_gen = test_generator.flow(test_images, test_oh_labels, batch_size=BATCH_SIZE, shuffle=False)
vgg_model.evaluate(flow_test_gen)

import matplotlib.pyplot as plt

def show_history(history):
    plt.figure(figsize=(8,4))
    plt.yticks(np.arange(0, 1, 0.05))
    plt.xticks(np.arrage(0, 30, 2))
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='valid')
    plt.legend()

show_history(history)
