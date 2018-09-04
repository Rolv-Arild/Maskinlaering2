import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class SigmoidModel:

    def __init__(self):
        # Model input
        self.x = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.float32)

        # Model variables
        self.W = tf.Variable([[0.0]])
        self.b = tf.Variable([[0.0]])

        # Logits
        logits = tf.matmul(self.x, self.W) + self.b

        # Predictor
        f = tf.sigmoid(logits)

        # Uses Cross Entropy
        self.loss = tf.losses.sigmoid_cross_entropy(self.y, logits)


model = SigmoidModel()

x_train = np.mat([[0], [1]])
y_train = np.mat([[1], [0]])

minimize_operation = tf.train.AdagradOptimizer(10).minimize(model.loss)

session = tf.Session()

session.run(tf.global_variables_initializer())

for epoch in range(10000):
    session.run(minimize_operation, {model.x: x_train, model.y: y_train})

W, b, loss = session.run([model.W, model.b, model.loss], {model.x: x_train, model.y: y_train})

print("W = %s, b = %s, loss = %s" % (W, b, loss))

session.close()


def f(t):
    return 1 / (1 + np.exp(-(t * W + b)))


fig, ax = plt.subplots()

ax.plot(x_train, y_train, 'o')
ax.set_xlabel('x')
ax.set_ylabel('y')

x = np.linspace(np.min(x_train), np.max(x_train), 1000)
y = np.transpose(f(x))
ax.plot(x, y)

ax.legend()
plt.show()
