import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

test_data = pd.read_csv('linearlyseparabletest.data', header=None).values

test_X = test_data[:, :-1]
test_Y = test_data[:, -1].reshape(-1, 1)


def calculate_probabilities(weights, bias, features):
    linear_score = np.dot(features, weights) + bias
    exp_term = np.exp(-linear_score)

    probability_positive_class = 1 / (1 + exp_term)
    probability_negative_class = exp_term / (1 + exp_term)

    return probability_positive_class, probability_negative_class, linear_score


def compute_gradients(weights, features, bias, labels, regularization_lambda):
    num_samples, num_features = features.shape
    prob_positive_class, _, functional_margin = calculate_probabilities(weights, bias, features)

    gradient_bias = ((0.5 * (labels + 1)) - prob_positive_class).sum()
    gradient_weights = np.sum(features * ((0.5 * (labels + 1)) - prob_positive_class), axis=0).reshape(
        (num_features, 1)) - regularization_lambda * weights

    likelihood = np.sum(0.5 * (labels + 1) * functional_margin - np.log(1 + np.exp(functional_margin))) - (
        regularization_lambda) * np.sum(weights ** 2)

    return gradient_bias, gradient_weights, likelihood


def calculate_loss(W, Y, X, bias):
    return np.sum(0.5 * (Y + 1) * (np.dot(X, W) + bias) - np.log(1 + np.exp(np.dot(X, W) + bias)))


def train_logistic_regression(features, labels, learning_rate, regularization_lambda):
    num_samples, num_features = features.shape
    weights = np.ones(num_features).reshape(num_features, 1) * 0.5
    bias = 0.5
    iterations = 0

    gradient_bias, gradient_weights, likelihood = compute_gradients(weights, features, bias, labels,
                                                                    regularization_lambda)
    current_likelihood = likelihood

    while True:
        previous_likelihood = current_likelihood
        weights = weights + learning_rate * gradient_weights
        bias = bias + learning_rate * gradient_bias
        iterations = iterations + 1
        gradient_bias, gradient_weights, current_likelihood = compute_gradients(weights, features, bias, labels,
                                                                                regularization_lambda)

        if abs(current_likelihood - previous_likelihood) < 1e-4:
            break

    return weights, bias, iterations


def calculate_accuracy(features, labels, weights, bias):
    num_samples, _ = features.shape
    predicted_scores = np.dot(features, weights) + bias
    predicted_labels = np.sign(predicted_scores)

    misclassifications = np.sum(np.abs(labels - predicted_labels)) / 2
    accuracy_percentage = (1 - misclassifications / num_samples) * 100

    return accuracy_percentage


def validate_logistic_regression(X_train, Y_train, X_val, Y_val):
    best_weights = best_bias = best_learning_rate = best_regularization_lambda = None
    best_accuracy = 0

    for i in range(-6, -5):
        learning_rate = 10 ** i

        for j in range(-4, 5):
            regularization_lambda = 10 ** j

            (weights, bias, _) = train_logistic_regression(X_train, Y_train, learning_rate, regularization_lambda)
            current_accuracy = calculate_accuracy(X_val, Y_val, weights, bias)

            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                best_weights = weights
                best_bias = bias
                best_learning_rate = learning_rate
                best_regularization_lambda = regularization_lambda

    return best_weights, best_bias, best_learning_rate, best_regularization_lambda


(W, bias, alpha, lamb) = validate_logistic_regression(test_X, test_Y, test_X, test_Y)

test_accuracy = calculate_accuracy(test_X, test_Y, W, bias)
print('Test Accuracy:  ', test_accuracy)
print('W: ', W)
print('bias: ', bias)
print('lambda: ', lamb)

plt.scatter(test_X[:, 0], test_X[:, 1], c=test_Y, cmap=plt.cm.Paired, edgecolors='k', marker='o')

# Plot the decision boundary
x_min, x_max = test_X[:, 0].min() - 1, test_X[:, 0].max() + 1
y_min, y_max = test_X[:, 1].min() - 1, test_X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
Z = np.dot(np.c_[xx.ravel(), yy.ravel()], W) + bias
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

plt.title('Logistic Regression Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()