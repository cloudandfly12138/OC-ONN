#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
'''
from datetime import datetime
import os
import tensorflow as tf
import numpy as np
from oc_onn import ONN
from oc_onn_data import ImageDataGenerator
import matplotlib.pyplot as plt

Iterator = tf.data.Iterator

# some params
batch_size = 6
num_classes = 10
learning_rate = 0.001
num_epochs = 10
# How often we want to write the tf.summary data to disk
display_step = 200
train_layers = ['conv5', 'conv4', 'conv3', 'conv2', 'conv1']
filewriter_path = 'log'
checkpoint_path = 'log'

# Create parent path if it doesn't exist
if not os.path.isdir(filewriter_path):
    os.mkdir(filewriter_path)
if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)

# load mnist data 
x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')

# Assume that each row of `images` corresponds to the same row as `labels`.
assert x_train.shape[0] == y_train.shape[0]
assert x_test.shape[0] == y_test.shape[0]


# with tf.device('/cpu:0'):
tr_data = ImageDataGenerator(   mode='train',
                                batch_size=batch_size,
                                num_classes=num_classes,
                                shuffle=True)

test_data = ImageDataGenerator( mode='test',
                                batch_size=batch_size,
                                num_classes=num_classes,
                                shuffle=False)
# create an reinitializable iterator given the dataset structure
iterator = Iterator.from_structure(tr_data.data.output_types,
                                    tr_data.data.output_shapes)
next_batch = iterator.get_next()


# Ops for initializing the two different iterators
train_init_op = iterator.make_initializer(tr_data.data)
test_init_op = iterator.make_initializer(test_data.data)

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, 100, 100])
y = tf.placeholder(tf.int32, [batch_size])
# keep_prob = tf.placeholder(tf.float32)

# our forward model
model = ONN(x, batch_size, num_classes)
# link variables to output
score = model.output
# List of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]


# Op for calculating the loss
with tf.name_scope("cross_ent"):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=score,
                                                                  labels=y))

# Train op
with tf.name_scope("train"):
    # Create optimizer and apply gradient descent to the trainable variables
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # Get gradients of all trainable variables
    gradients = optimizer.compute_gradients(loss, var_list)

    train_op = optimizer.apply_gradients(grads_and_vars=gradients)

# Add gradients to summary
for gradient, var in gradients:
    tf.summary.histogram(var.name + '/gradient', gradient)

# Add the variables we train to the summary
for var in var_list:
    tf.summary.histogram(var.name, var)

# Add the loss to summary
tf.summary.scalar('cross_entropy', loss)


# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.cast(tf.argmax(score, 1),tf.int32), y)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Add the accuracy to the summary
tf.summary.scalar('accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Get the number of training/validation steps per epoch
train_batches_per_epoch = int(np.floor(tr_data.data_size/batch_size))
test_batches_per_epoch = int(np.floor(test_data.data_size/batch_size))
# # 测试
# train_batches_per_epoch = 100
# test_batches_per_epoch = 100

# Start Tensorflow session
with tf.Session() as sess:
    # Initialize all variables
    sess.run(tf.global_variables_initializer())
#    saver = tf.train.Saver()
#    saver.restore(sess,'/path/to/checkpoints')
    # Add the model graph to TensorBoard
    writer.add_graph(sess.graph)


    # test before trainng
    sess.run(test_init_op, feed_dict={  test_data.x_placeholder: x_test,
                                        test_data.y_placeholder: y_test  })
    test_acc = 0.
    test_count = 0
    for _ in range(test_batches_per_epoch):
        img_batch, label_batch = sess.run(next_batch)
        acc = sess.run(accuracy, feed_dict={x: img_batch,
                                            y: label_batch})
        test_acc += acc
        test_count += 1
#            print(test_count)
    test_acc /= test_count
    print("{} Before Training, Test Accuracy = {:.4f}".format(datetime.now(),
                                                       test_acc))



    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                      filewriter_path))
    # Loop over number of epochs
    for epoch in range(num_epochs):

        print("{} Epoch number: {}".format(datetime.now(), epoch+1))

        # Initialize iterator with the training dataset
        sess.run(train_init_op, feed_dict={ tr_data.x_placeholder: x_train,
                                            tr_data.y_placeholder: y_train  })
        for step in range(train_batches_per_epoch):
            # if step+1 % 1000 == 0:
            #     print(step+1)
            # get next batch of data
            img_batch, label_batch = sess.run(next_batch)

            # And run the training op
            sess.run(train_op, feed_dict={x: img_batch,
                                          y: label_batch})
            if step % display_step == 0:
                s = sess.run(merged_summary, feed_dict={x: img_batch,
                                                        y: label_batch})
                writer.add_summary(s, epoch*train_batches_per_epoch + step)



        # Validate the model on the entire test set
        print("{} Start Testing".format(datetime.now()))
        sess.run(test_init_op, feed_dict={  test_data.x_placeholder: x_test,
                                            test_data.y_placeholder: y_test  })
        test_acc = 0.
        test_count = 0

        for _ in range(test_batches_per_epoch):

            img_batch, label_batch = sess.run(next_batch)
            acc = sess.run(accuracy, feed_dict={x: img_batch,
                                                y: label_batch})
            test_acc += acc
            test_count += 1
    #            print(test_count)
        test_acc /= test_count
        print("{} Test Accuracy = {:.4f}".format(datetime.now(),
                                                       test_acc))

        # checkpoint after each epoch
        checkpoint_name = os.path.join(checkpoint_path,
                                       'model_epoch'+str(epoch+1)+'.ckpt')
        save_path = saver.save(sess, checkpoint_name)
        print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                       checkpoint_name))
    

    # show onn.intensity of test_data's last-batch  
    # %matplotlib inline
    plt.figure(figsize=(10,10))
    for i in range(batch_size):
        plt.subplot(2,batch_size,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(img_batch[i], cmap=plt.cm.binary)
        plt.xlabel(label_batch[i])
        # plt.title('image%d'%i)

        plt.subplot(2,batch_size, batch_size+i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        inten = sess.run(model.intensity,feed_dict={x: img_batch,
                                                     y: label_batch})
        plt.imshow(inten[i],cmap=plt.cm.binary)
        plt.xlabel(label_batch[i])
    plt.show()

