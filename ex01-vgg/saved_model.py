import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split

vgg_model = tf.keras.models.load_model('/home/bginess/tf-examples/vgg/my_model')

vgg_model.summary()


BATCH_SIZE = 64


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

from tensorflow.keras.datasets import cifar10

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

test_generator = ImageDataGenerator(rescale=1/255.0)

(tr_images, tr_oh_labels), (val_images, val_oh_labels), (test_images, test_oh_labels) = get_train_vaild_test_set(train_images, train_labels, test_images, test_labels, valid_size=0.15, random_state=2021)

flow_test_gen = test_generator.flow(test_images, test_oh_labels, batch_size=BATCH_SIZE, shuffle=False)
vgg_model.evaluate(flow_test_gen)

