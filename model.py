#!/usr/bin/env python
"""
Steering angle prediction model
"""

import image_processing as ip
import csv
import argparse
import json
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten, Lambda
from keras.layers.convolutional import Convolution2D
import cv2
import numpy as np
import random

ch, row, col = 3, 60, 320  # cropped camera format


def load_data(file_name, split):
    """
    Splits data into a validation and training set (the training set is
    not actually processed - it is a list of driving entries yet to import)
    :param file_name: csv file containing driving data
                      Expected format is center, left, right, angle, throttle, breaking, speed
    :param split: size of the validation split
    :return: csv data for training, validation data, validation labels
    """
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        csv_data = list(reader)

    val_size = int(len(csv_data) * split)
    print("Data set has {0} entries, validation set size = {1}".format(len(csv_data), val_size))
    random.seed(123)
    random.shuffle(csv_data)

    x = []
    y = []
    for entry in csv_data[:val_size]:
        center_img = entry[0]
        angle = entry[3]
        x.append(ip.normalize_image(cv2.imread(center_img, cv2.IMREAD_COLOR)))
        y.append(angle)
    return csv_data[val_size:], np.asarray(x), np.asarray(y)


def generate_batch(csv_data, batch_size):
    steering_adj = 0.2
    steering_off = 0.27
    while 1:
        num_entries = len(csv_data)
        for i in range(0, num_entries, batch_size):
            x = []
            y = []
            for entry in csv_data[i: i + batch_size]:
                sel = np.random.randint(0, 4)
                center_img = cv2.imread(entry[0], cv2.IMREAD_COLOR)
                left_img = cv2.imread(entry[1], cv2.IMREAD_COLOR)
                right_img = cv2.imread(entry[2], cv2.IMREAD_COLOR)
                angle = entry[3]

                # Images are preprocessed already, so all we have to do is
                # to apply the random brightness adjust
                if sel == 0:
                    x.append(ip.random_brightness_adjust(center_img))
                    y.append(float(angle))
                elif sel == 1:
                    x.append(ip.random_brightness_adjust(left_img))
                    l_angle = float(angle) + steering_off + (abs(float(angle)) * steering_adj)
                    y.append(l_angle)
                elif sel == 2:
                    x.append(ip.random_brightness_adjust(right_img))
                    r_angle = float(angle) - steering_off - (abs(float(angle)) * steering_adj)
                    y.append(r_angle)
                else:
                    # Flipping (only for center image)
                    x.append(ip.random_brightness_adjust(cv2.flip(center_img, 1)))
                    y.append(-float(angle))
            yield np.asarray(x), np.asarray(y)


def get_model():
    """ Inspired on NVidia's model, with one less convolutional layer"""
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
              input_shape=(row, col, ch),
              output_shape=(row, col, ch)))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="same", activation="relu"))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="same", activation="relu"))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="same", activation="relu"))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="same", activation="relu"))
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Dropout(.5))
    model.add(Activation("relu"))
    model.add(Dense(100))
    model.add(Dropout(.2))
    model.add(Activation("relu"))
    model.add(Dense(50))
    model.add(Activation("relu"))
    model.add(Dense(10))
    model.add(Activation("relu"))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Steering angle model trainer')
    parser.add_argument('--data', default='./driving_log.csv', help='Path to CSV file with input data.')
    parser.add_argument('--val_split', type=float, default=0.3, help='Validation set split size.')
    parser.add_argument('--batch', type=int, default=64, help='Batch size.')
    parser.add_argument('--epoch', type=int, default=20, help='Number of epochs.')
    parser.add_argument('--epochsize', type=int, default=50000, help='How many frames per epoch.')
    args = parser.parse_args()

    train_set, x_val, y_val = load_data(args.data, args.val_split)
    val_tuple = (x_val, y_val) if args.val_split > 0 else None
    epochsize = min(len(train_set), args.epochsize)
    print("Training set size = {0}, epoch size = {1}".format(len(train_set), epochsize))
    model = get_model()
    model.summary()
    model.fit_generator(
      generate_batch(train_set, args.batch),
      samples_per_epoch=epochsize,
      nb_epoch=args.epoch,
      validation_data=val_tuple,
      nb_val_samples=(args.epochsize // 5)
    )
    print("Saving model weights and configuration file.")

    model.save_weights('./model.h5')
    with open('./model.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)
