import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

data = np.loadtxt('ex1data1.txt', unpack=True, dtype=np.float32, delimiter=',')
data = data.transpose()
# data shape (97,2)

# note : x*theta dim == y dim
x = np.concatenate((np.ones((len(data), 1)), np.reshape(data[:, 0], (len(data), 1))), axis=1)
print(x)

y = np.reshape(data[:, 1], (len(data), 1))
theta = tf.Variable(tf.random_uniform([2, 1], -1.0, 1.0))
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = tf.matmul(X, theta)
cost_function = tf.reduce_mean(tf.square(hypothesis-Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost_function)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(10000):
        sess.run([optimizer, cost_function], feed_dict={X:x, Y:y})
        if i % 1000 == 0:
            print(sess.run([cost_function, theta], feed_dict={X:x, Y:y}))

    theta_value = sess.run(theta, feed_dict={X:x, Y:y})
    print (theta_value)

# plotting the data
plt.plot(x[:, 1], y, 'bo', label='data')
plt.plot(x[:, 1], x[:, 1]*theta_value[1, 0] + theta_value[0, 0], 'r', label='predicted line')
plt.legend()
plt.show()