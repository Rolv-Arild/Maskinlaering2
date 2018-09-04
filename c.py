import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, art3d


class SigmoidModel:

    def __init__(self):
        # Model input
        self.x = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.float32)

        # Model variables
        self.W1 = tf.Variable(tf.random_normal([2, 2]))
        self.b1 = tf.Variable(tf.zeros([2]))

        self.W2 = tf.Variable(tf.random_normal([2, 1]))
        self.b2 = tf.Variable(tf.zeros([1]))

        f1 = tf.sigmoid(tf.add(tf.matmul(self.x, self.W1), self.b1))
        logits_f2 = tf.add(tf.matmul(f1, self.W2), self.b2)

        self.f2 = tf.sigmoid(logits_f2)

        self.loss = tf.losses.sigmoid_cross_entropy(self.y, logits_f2)


model = SigmoidModel()

x_train = np.mat([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.mat([[0], [1], [1], [0]])

minimize_operation = tf.train.AdagradOptimizer(0.1).minimize(model.loss)

session = tf.Session()

session.run(tf.global_variables_initializer())

for epoch in range(10000):
    session.run(minimize_operation, {model.x: x_train, model.y: y_train})

W1, b1, W2, b2, loss = session.run([model.W1, model.b1, model.W2, model.b2, model.loss],
                                   {model.x: x_train, model.y: y_train})

print("W1 = %s, b1 = %s, W2 = %s, b2 = %s, loss = %s" % (W1, b1, W2, b2, loss))

session.close()


def sigmoid(t):
    return 1 / (1 + np.exp(-t))


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X = np.mat([x_train.item(i, 0) for i in range(0, len(x_train))])
Y = np.mat([x_train.item(i, 1) for i in range(0, len(x_train))])
Z = np.mat([y_train.item(i, 0) for i in range(0, len(y_train))])

ax.scatter(X, Y, Z, c='r', marker='o')

# Observed/training input and output

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.set_zlim(0)

size = 100

x = np.linspace(0, 1, size)
y = np.linspace(0, 1, size)

X, Y = np.meshgrid(x, y)

session = tf.Session()

session.run(tf.global_variables_initializer())

# x = np.transpose([x, y])
Z = np.ndarray([size, size])
for i in range(size):
    for j in range(size):
        Z[i][j] = session.run(model.f2, {model.x: [[x[i], y[j]]], model.W1: W1, model.W2: W2, model.b1: b1, model.b2: b2})

surf = ax.plot_surface(X, Y, Z, shade=False, alpha=0.5)

ax.legend()
plt.show()
