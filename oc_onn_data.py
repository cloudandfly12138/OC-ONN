#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
'''
import tensorflow as tf
import numpy as np
Dataset = tf.data.Dataset

class ImageDataGenerator(object):
    """
    Wrapper class around the new Tensorflows dataset pipeline.
    """

    def __init__(self, mode, batch_size, num_classes, shuffle=True):
        '''
        Create a new ImageDataGenerator.

        Args:
            x_npy_file: Path to the x_npy file. shape:(data_size,100,100) 
                dtype:'float32' value:[0,1]
            y_npy_file: Path to the y_npy file. shape:(data_size,) 
                dtype:'uint8' value:0~9
            mode: Either 'training' or 'inference'. Depending on this value,
                different parsing functions will be used.
            batch_size: Number of images per batch.
            num_classes: Number of classes in the dataset.
            shuffle: Wether or not to shuffle the data in the dataset.
            buffer_size: Number of images used as buffer for TensorFlows
                shuffling of the dataset.

        Raises:
            ValueError: If an invalid mode is passed.

        '''
        self.num_classes = num_classes
        self.batch_size = batch_size

        if mode == 'train':
            data_size = 60000
        elif mode == 'test':
            data_size = 10000
        else:
            raise ValueError("Invalid mode '%s'." % (mode))
        # number of samples in the dataset
        self.data_size = data_size

        self.x_placeholder = tf.placeholder(tf.float32, [data_size,100,100])
        self.y_placeholder = tf.placeholder(tf.int32, [data_size])

        # create dataset
        data = Dataset.from_tensor_slices((self.x_placeholder, self.y_placeholder))

        # shuffle the `buffer_size` elements of the dataset
        if shuffle:
            data = data.shuffle(buffer_size = self.data_size)

        # create a new dataset with batches of images
        data = data.batch(batch_size)

        self.data = data
