import pandas as pd
import numpy as np
import cvxopt
import matplotlib.pyplot as plt

C = [.001, .01, .1, 1, 10, 100, 1000]

test_data = np.array(pd.read_csv('linearlyseparabletest.data', header=None))
testX = test_data[:, :-1]
testY = test_data[:, -1]
testY[testY == 0] = -1

row, col = test_data.shape


def calculate_accuracy(x, y, w, b):
    rows = len(x)
    count = 0
    for i in range(rows):
        pred = np.dot(w[:len(x[i])], x[i]) + b
        if pred * y[i] > 0:
            count += 1

    pred_accuracy = (count / rows) * 100
    return pred_accuracy


P = np.eye(row + col + 1)
P[:col, :col] = 1.0
P = cvxopt.matrix(P)

h = np.vstack((-np.ones((row, 1)), np.zeros((row, 1))))
h = cvxopt.matrix(h)

G = np.zeros((2 * row, row + col + 1))
for i in range(row):
    for j in range(testX.shape[1]):
        G[i][j] = -1 * testY[i] * testX[i][j]

    G[i][col + i] = -1
    G[i][row + col] = -1 * testY[i]
    G[row + i][col + i] = -1

G = cvxopt.matrix(G)

for c in C:
    Q = np.zeros((row + col + 1, 1))
    Q[col:row + col, 0] = c
    q = cvxopt.matrix(Q)

    result = cvxopt.solvers.qp(P, q, G, h)

    optimal_sol = result['x']

    w = []
    for i in range(len(optimal_sol)):
        w.append(optimal_sol[i])

    b = optimal_sol[row + col]

    accuracy = calculate_accuracy(testX, testY, w, b)
    print('Test accuracy: ', accuracy, ' for value of C =', c)

plt.scatter(testX[:, 0], testX[:, 1], c=testY, cmap=plt.cm.Paired, edgecolors='k', marker='o')

# Plot the decision boundary
x_min, x_max = testX[:, 0].min() - 1, testX[:, 0].max() + 1
y_min, y_max = testX[:, 1].min() - 1, testX[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
Z = np.dot(np.c_[xx.ravel(), yy.ravel()], w[:2]) + b
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

plt.title('SVM Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()