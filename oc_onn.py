#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
one-channel onn with five conv_op layers
'''

import tensorflow as tf
import numpy as np


# xx, yy, Lambda, k_z_values can all be generated from onn_run.ipynb.
xx = np.load('xx.npy')
yy = np.load('yy.npy')
Lambda = np.load('Lambda.npy')
k_z = np.load('k_z.npy')

x_tensor = tf.constant(xx, tf.float32)
y_tensor = tf.constant(yy, tf.float32)
Lambda_tensor = tf.constant(Lambda, tf.float32)
k_z_tensor = tf.constant(k_z, tf.complex64)
f = tf.constant(5E-2, tf.float32)


def fftshift_tf(data):
    """
    :param data: input tensor to do fftshift
    :return: after fftshift
    """
    dims = tf.shape(data)
    num = dims[1]
    shift_amt = (num - 1) / 2
    shift_amt = tf.cast(shift_amt, tf.int32)
    output = tf.manip.roll(data, shift=shift_amt, axis=0)
    output = tf.manip.roll(output, shift=shift_amt, axis=1)

    return output


def ifftshift_tf(data):
    """
    Performs an ifftshift operation on the last two dimensions of a 4-D input tensor
    :param data: input tensor to do ifftshift
    :return: after ifftshift
    """
    dims = tf.shape(data)
    num = dims[1]
    shift_amt = (num + 1) / 2
    shift_amt = tf.cast(shift_amt, tf.int32)
    output = tf.manip.roll(data, shift=shift_amt, axis=0)
    output = tf.manip.roll(output, shift=shift_amt, axis=1)

    return output


def generate_phase():
    """
    Generates the phase for a lens based on the focal length variable "f".
    Other referenced variables are global
    Return: phase generated
    """
    phase = tf.constant(2 * np.pi, tf.float32)\
            / Lambda_tensor * (tf.sqrt(tf.square(x_tensor) + tf.square(y_tensor) + tf.square(f)) - f)
    phase = tf.cast(phase, tf.complex64)
    return phase


def generate_propagator():
    """
    Generates the Fourier space propagator based on the focal length variable "f".
    Other referenced variables are global. 
    Actually, it's the ``Transfer function of angular spectrum theory``

    Return: propagator generated
    """
    propagator = tf.exp(1j * k_z_tensor * tf.cast(f, tf.complex64))
    propagator = ifftshift_tf(propagator)

    return propagator


def propagate(input_field, propagator):
    """
    Propagate an input E-field distribution along the optical axis using the defined propagator
    :param input_field: input field for doing propagation
    :param propagator: generated propagator
    :return: result after propagation
    """
    output = tf.ifft2d(tf.fft2d(input_field) * propagator)

    return output


def simulate_4f_system(input_field, mask):
    """
    Pass an image through a 4f system
    Args:
      input_field: A `tensor` type of `complex64` with batch_size dimension
      mask: Transmission coefficient of mask, fft_2d(kernel), with batch_size dimension
    Return: 
      output of our 4f system, shape: [batch_size, 100,100]
    """
    # Calculate the lens phase
    lens_phase = generate_phase()

    # Calculate the propagator
    propagator = generate_propagator()

    # Propagate up to the first lens
    before_l1 = propagate(input_field, propagator)

    # Apply lens1 and propagate to the filter plane
    before_mask = propagate(before_l1 * tf.keras.backend.exp(-1j * lens_phase), propagator)

    # Apply kernel and propagate to the second lens
    before_l2 = propagate(before_mask * mask, propagator)

    # Apply lens2 and propagate to the output plane
    output = propagate(before_l2 * tf.keras.backend.exp(-1j * lens_phase), propagator)

    # Return output of the 4f optical convolution
    return output


def convolve_op(image, batch_size, name):
    """
    doing convolution  in frequency domain
    Args:
      image: A `tensor` type of `complex64`,input image or output of previous conv
      kernel_in: kernel for doing convolution
      name: scope name
    Return:
      result after doing convolution_op
    """
    with tf.variable_scope(name) :
        kernel = tf.get_variable(name='weights', trainable=True,
                                 shape=[10,10])
    #         f = tf.get_variable(name ='f', initializer=0.3E-2, trainable=True)
    # Zero pad the kernels for subsequent Fourier processing
    kernels = tf.concat([kernel, tf.constant(np.zeros((10,90)), tf.float32)], axis=1)
    kernels = tf.concat([kernels, tf.constant(np.zeros((90,100)), tf.float32)], axis=0)

    # Align the kernels for Fourier transforming
    # kernels = tf.transpose(kernels, perm=[3, 2, 0, 1])
    kernels = tf.cast(kernels, tf.complex64)
    mask = tf.fft2d(kernels)
    mask = ifftshift_tf(mask)

    # # Add an extra dimension for the batch size and duplicate
    # # the kernels to apply equally to all images in the batch
    # mask = tf.expand_dims(mask, axis=0)
    # mask = tf.tile(mask, multiples=[batch_size, 1, 1])

    output = simulate_4f_system(image, mask)

    return output


class ONN(object):
    """
    """

    def __init__(self, x, batch_size, num_classes):
        """
        Create the graph of the ONN model.
        Args:
            x: Placeholder for the input tensor.
            batch_size: batch_size for training
            num_classes: Number of classes in the dataset.
        """
        # Parse input arguments into class variables
        x = tf.cast(x, tf.complex64)
        self.X = x
        self.NUM_CLASSES = num_classes
        self.BATCH_SIZE = batch_size

        # Call the create function to build the computational graph of our network
        self.create()

    def create(self):
        """Create the network graph."""
        # 5 Layers: OP-conv
        conv1 = convolve_op(self.X, self.BATCH_SIZE, 'conv1')
        conv2 = convolve_op(conv1, self.BATCH_SIZE, 'conv2')
        conv3 = convolve_op(conv2, self.BATCH_SIZE, 'conv3')
        conv4 = convolve_op(conv3, self.BATCH_SIZE, 'conv4')
        conv5 = convolve_op(conv4, self.BATCH_SIZE, 'conv5')
		
        self.output = self.detector(conv5)

    def detector(self, input):
        '''
        Detecting the input field using 10 detector with fixed position
        Args:
            input: A `tensor`, shape:[batch_size,100,100], 
                the field before detectors.
        Return:
            A `tensor`, shape:[batch_size, num_classes]
                the intensity detected.
        '''
        # detect the intensity(proportional to the square of the electrial field)
        intensity = tf.abs(input)**2
        self.intensity = intensity
        
		# intensity 0~9
        in_0 = tf.reduce_sum(intensity[:, 12:22, 12:22], [1,2])
        in_1 = tf.reduce_sum(intensity[:, 12:22, 45:55], [1,2])
        in_2 = tf.reduce_sum(intensity[:, 12:22, 78:88], [1,2])

        in_3 = tf.reduce_sum(intensity[:, 45:55, 8:18], [1,2])
        in_4 = tf.reduce_sum(intensity[:, 45:55, 32:42], [1,2])
        in_5 = tf.reduce_sum(intensity[:, 45:55, 56:66], [1,2])
        in_6 = tf.reduce_sum(intensity[:, 45:55, 80:90], [1,2])

        in_7 = tf.reduce_sum(intensity[:, 78:88, 12:22], [1,2])
        in_8 = tf.reduce_sum(intensity[:, 78:88, 45:55], [1,2])
        in_9 = tf.reduce_sum(intensity[:, 78:88, 78:88], [1,2])
        
		# output shape:[batch_size, 10]
        output = tf.stack( [in_0, in_1, in_2, 
							in_3, in_4, in_5, in_6,
							in_7, in_8, in_9], 
							axis = 1)
							

        return output
