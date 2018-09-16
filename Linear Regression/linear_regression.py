import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

W_ = 0.1 # Weight
b_ = 0.3 # baise

# generate 1000 random points around the line.
num_points = 1000
vector_set = []
for i in range(num_points):
    x1 = np.random.normal(0.0, 0.55)
    y1 = x1 * W_ + b_ + np.random.normal(0.0, 0.03)
    vector_set.append([x1, y1])

x_data = [v[0] for v in vector_set]
y_data = [v[1] for v in vector_set]

plt.scatter(x_data, y_data, c='r')
plt.show()

W = tf.Variable(tf.random.uniform([1], -1.0, 1.0), name='W')
b = tf.Variable(tf.zeros([1]), name='bug')
y = W * x + b

loss = tf.reduce_mean(tf.square(y - y_data), name='loss')
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss, name='train')

with tf.Session() as sess:
