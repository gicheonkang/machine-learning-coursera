import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# load & data settings
data = np.loadtxt('ex1data1.txt', unpack=True, dtype=np.float32, delimiter=',')
data = data.transpose()

# It is easy to calculate X, Y training data is vector
Xdata = data[:, 0]
Ydata = data[:, 1]

# placeholder for numpy array to tensorflow array !
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# theta(weight) and bias
theta = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
bias = tf.Variable(0.0)
hypothesis = X*theta + bias

cost_function = tf.reduce_mean(tf.square(hypothesis-Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost_function)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(50000):
        sess.run([optimizer, cost_function], feed_dict={X: Xdata, Y: Ydata})
        if i % 1000 == 0:
            print(sess.run([cost_function, theta], feed_dict={X: Xdata, Y: Ydata}))

    wb, bb = sess.run([theta, bias], feed_dict={X:Xdata, Y:Ydata})
    print('theta :', wb)
    print('bias : ', bb)

# plotting
XX, YY = Xdata.T, Ydata.T
plt.plot(XX, YY, 'bo', label="data")
plt.plot(XX, XX*wb+bb, 'r', label='predicted')
plt.legend()
plt.show()
