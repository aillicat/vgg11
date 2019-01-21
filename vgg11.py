from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
import os
from keras.datasets import mnist
import pickle
from keras.utils import np_utils

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, History

script_dir = os.path.dirname(os.path.realpath(__file__))


def vgg11(x):
    # Creates a VGG11 model

    model = Sequential()
    model.add(
        Convolution2D(
            64, (3, 3),
            padding='same',
            input_shape=x.shape[1:],
            activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Convolution2D(256, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(512, (3, 3), padding='same', activation='relu'))
    model.add(Convolution2D(512, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(512, (3, 3), padding='same', activation='relu'))
    model.add(Convolution2D(512, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    return model


def load_data():
    # This function resizes the image from 28x28 to 32x32 so pooling will not run into error
    # Outputs resized data file to be loaded later

    (x_train_pre, y_train), (x_test_pre, y_test) = mnist.load_data()

    x_train = np.zeros((x_train_pre.shape[0], 32, 32))

    for i in range(0, x_train.shape[0]):
        img = Image.fromarray(x_train_pre[i, :]).resize((32, 32))
        x_train[i, :] = np.asarray(img, 'float32')

    x_test = np.zeros((x_test_pre.shape[0], 32, 32))

    for i in range(0, x_test.shape[0]):
        img = Image.fromarray(x_test_pre[i, :]).resize((32, 32))
        x_test[i, :] = np.asarray(img, 'float32')

    with open(os.path.join(script_dir, 'MNISTdata'), 'wb') as f:
        pickle.dump(x_train, f)
        pickle.dump(y_train, f)
        pickle.dump(x_test, f)
        pickle.dump(y_test, f)


if __name__ == '__main__':

    load_data()

    with open(os.path.join(script_dir, 'MNISTdata'), 'rb') as f:
        x_train = pickle.load(f)
        y_train = pickle.load(f)
        x_test = pickle.load(f)
        y_test = pickle.load(f)

    # use this to display some images
    # img = Image.fromarray(x_train[1,:])
    # Image._show(img)

    x_train = x_train.reshape(x_train.shape[0], 32, 32, 1)
    x_test = x_test.reshape(x_test.shape[0], 32, 32, 1)
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    weight_path = os.path.join(script_dir, 'vgg11_weights.hdf5')

    model = vgg11(x_train)
    sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)

    # model.load_weights(weight_path)
    model.compile(
        optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
    )

    datagen.fit(x_train)

    # model.load_weights(weight_path)
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001)
    checkpoint = ModelCheckpoint(
        weight_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode='min')
    history = model.fit_generator(
        datagen.flow(x_train, y_train, batch_size=256),
        steps_per_epoch=x_train.shape[0] // 256,
        validation_data=(x_test, y_test),
        epochs=74,
        callbacks=[reduce_lr, checkpoint])

    # Summarize history for accuracy of first 5 epochs
    plt.plot(history.history['acc'][0:5])
    plt.title('Train Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()

    plt.plot(history.history['val_acc'][0:5])
    plt.title('Test Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()

    # Summarize history for loss of first 5 epochs
    plt.plot(history.history['loss'][0:5])
    plt.title('Train Loss')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()

    plt.plot(history.history['val_loss'][0:5])
    plt.title('Test Loss')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()
