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
        self.W = tf.Variable([[0.0], [0.0]])
        self.b = tf.Variable([[0.0]])

        # Logits
        logits = tf.matmul(self.x, self.W) + self.b

        # Predictor
        f = tf.sigmoid(logits)

        # Uses Cross Entropy
        self.loss = tf.losses.sigmoid_cross_entropy(self.y, logits)


model = SigmoidModel()

x_train = np.mat([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.mat([[1], [1], [1], [0]])

minimize_operation = tf.train.AdagradOptimizer(10).minimize(model.loss)

session = tf.Session()

session.run(tf.global_variables_initializer())

for epoch in range(10000):
    session.run(minimize_operation, {model.x: x_train, model.y: y_train})

W, b, loss = session.run([model.W, model.b, model.loss], {model.x: x_train, model.y: y_train})

print("W = %s, b = %s, loss = %s" % (W, b, loss))

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

x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)

X, Y = np.meshgrid(x, y)

Z = sigmoid(W[0] * X + W[1] * Y + b)

surf = ax.plot_surface(X, Y, Z, shade=False, alpha=0.5)

ax.legend()
plt.show()
