import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Load observations from the mnist dataset. The observations are divided into a training set and a test set
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Show x of first observation in the training set
# plt.imshow(x_train[0, :])

# Print the classification of the first observation in the training set
# print(y_train[0])

# Save x of first observation in the training set
# plt.imsave('x_train_1.png', x_train[0, :])

# plt.show()

x_train = np.reshape(x_train, [60000, 784])
y_train = tf.keras.utils.to_categorical(y_train, 10)


# print(x_train)

class SoftmaxModel:

    def __init__(self):
        # Model input
        self.x = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.float32)

        # Model variables
        self.W = tf.Variable(tf.random_normal([784, 10]))
        self.b = tf.Variable(tf.zeros([10]))

        logits = tf.add(tf.matmul(self.x, self.W), self.b)

        # Predictor
        self.f = tf.nn.softmax(logits)

        self.loss = tf.losses.softmax_cross_entropy(self.y, logits)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.f, 1), tf.argmax(self.y, 1)), tf.float32))


model = SoftmaxModel()

minimize_operation = tf.train.AdagradOptimizer(10).minimize(model.loss)

session = tf.Session()

session.run(tf.global_variables_initializer())

for epoch in range(50):
    session.run(minimize_operation, {model.x: x_train, model.y: y_train})

W, b, loss = session.run([model.W, model.b, model.loss],
                         {model.x: x_train, model.y: y_train})

x_test = np.reshape(x_test, [10000, 784])
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Find accuracy (how many % correct)
accuracy = session.run(model.accuracy, {model.x: x_test, model.y: y_test})

print("W = %s, b = %s, loss = %s" % (W, b, loss))
print("Accuracy: %s" % accuracy)

session.close()

W = np.transpose(W)

# Getting images from W
w = 28
h = 28
fig = plt.figure(figsize=(25, 10))
columns = 5
rows = 2

for i in range(columns * rows):

    # Show the images
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(np.reshape(W[i], [28, 28]))

    # Save the images
    plt.imsave('W%d.png' % i, np.reshape(W[i], [28, 28]))


plt.show()
