import tensorflow as tf
from tensorflow.contrib.layers import flatten
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import random
import numpy as np
import matplotlib.pyplot as plt

# LeNet Neural Network
def LeNet(x):
    # Hyperparameters
    mu = 0
    sigma = 0.1

    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma), name='conv1_W')
    conv1_b = tf.Variable(tf.zeros(6), name='conv1_b')
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma), name='conv2_W')
    conv2_b = tf.Variable(tf.zeros(16), name='conv2_b')
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)

    # Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma), name='fc1_W')
    fc1_b = tf.Variable(tf.zeros(120), name='fc1_b')
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b

    # Activation.
    fc1    = tf.nn.relu(fc1)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma), name='fc2_W')
    fc2_b  = tf.Variable(tf.zeros(84), name='fc2_b')
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b

    # Activation.
    fc2    = tf.nn.relu(fc2)

    # Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 10), mean = mu, stddev = sigma), name='fc3_W')
    fc3_b  = tf.Variable(tf.zeros(10), name='fc3_b')
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits

# get original image (X_test[0])
mnist = input_data.read_data_sets("./LeNet/MNIST_data/", reshape=False)
X_test = np.pad(mnist.test.images[:1], ((0,0),(2,2),(2,2),(0,0)), 'constant')

# training
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 10)
rate = 0.001
logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
grads_and_vars = optimizer.compute_gradients(loss, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
training_operation = optimizer.apply_gradients(grads_and_vars)

# load model
saver = tf.train.Saver(max_to_keep=0)

tf.reset_default_graph()
lenet_model = tf.train.latest_checkpoint('./LeNet/ckpt')

# set variables and placeholders
image = tf.Variable(tf.zeros(shape=(1, 32, 32, 1)), name='modified_image')
wrong_label = tf.placeholder(tf.int32, (None),name="wrong_label")
new_saver = tf.train.import_meta_graph("./LeNet/ckpt/lenet-0.meta",
                input_map={"input":tf.convert_to_tensor(image),"output":wrong_label})
logits,loss = tf.get_collection("important_tensors")
logits=tf.to_float(logits)
loss = tf.to_float(loss)
new_optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001)

original_image = np.reshape(X_test[0],[32,32])

# max amount to change original image
max_change_above = original_image + 0.145
max_change_below = original_image - 0.145

# instance of original image
I = np.reshape(X_test[0],(1,32,32,1)) # this digit is a 7
new_init = tf.assign(image,I)
gvs = new_optimizer.compute_gradients(loss, [image])
new_training = optimizer.apply_gradients(gvs)

# train image to trick lenet
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, tf.train.latest_checkpoint('./LeNet/ckpt'))
    sess.run(new_init)
    logits_val_initial = logits.eval()
    print("logits before: ", logits_val_initial)
    print("chosen class of original image: ", np.argmax(logits_val_initial,axis=1))
    # number of times image will be modified
    for i in range(200):
        sess.run([new_training],feed_dict={wrong_label:0})
        # make changes to image according to loss function
        modified_image = np.reshape(image.eval(),[32,32])
        # clip back pixles that were changes too much
        modified_image = np.clip(modified_image, max_change_below, max_change_above)
        modified_image = np.clip(modified_image, -1.0, 1.0)
        I = np.reshape(modified_image,(1,32,32,1))
        # assin newly modified image to I
        new_init = tf.assign(image,I)
        if (i)%50==0:
          loss_val = loss.eval(feed_dict={wrong_label:0})
          print("Loss value at iteration %d: %f" % (i, loss_val))

    logits_val = logits.eval()
    print("logits after: ", logits_val)
    print("chosen class of modified image: ", np.argmax(logits_val,axis=1))

# save modified image
import scipy.misc
scipy.misc.imsave('modified.jpg',np.reshape(modified_image,[32,32]))

# compute PSNR between original and modified image
mse = np.mean((np.reshape(X_test[0],[32,32])-np.reshape(modified_image,[32,32]))**2)
psnr = -10.0*np.log10(mse)
print("PSNR between original and modified image: ", psnr)
