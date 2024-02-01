import pandas as pd
import numpy as np

training_data = pd.read_csv('sonar_train.data', header=None).values
validation_data = pd.read_csv('sonar_valid.data', header=None).values
test_data = pd.read_csv('sonar_test.data', header=None).values

training_data[training_data[:, -1] == 2, -1] = -1
validation_data[validation_data[:, -1] == 2, -1] = -1
test_data[test_data[:, -1] == 2, -1] = -1

train_X = training_data[:, :-1]
train_Y = training_data[:, -1].reshape(-1, 1)

val_X = validation_data[:, :-1]
val_Y = validation_data[:, -1].reshape(-1, 1)

test_X = test_data[:, :-1]
test_Y = test_data[:, -1].reshape(-1, 1)

np.random.seed(42)

def calculate_probabilities(weights, bias, features):
    linear_score = np.dot(features, weights) + bias
    exp_term = np.exp(-linear_score)

    probability_positive_class = 1 / (1 + exp_term)
    probability_negative_class = exp_term / (1 + exp_term)

    return probability_positive_class, probability_negative_class, linear_score


def calculate_gradients(W, X, bias, Y, lamb):
    M, N = X.shape
    prob_positive_class, _, functional_margin = calculate_probabilities(W, bias, X)
    gradient_b = ((0.5 * (Y + 1)) - prob_positive_class).sum()
    norm_derivative = W
    norm_derivative[norm_derivative < 0] = -1
    norm_derivative[norm_derivative >= 0] = 1
    gradient_w = np.sum(X * ((0.5 * (Y + 1)) - prob_positive_class), axis=0).reshape(N, 1) - lamb * norm_derivative
    likelihood = np.sum(0.5 * (Y + 1) * functional_margin - np.log(1 + np.exp(functional_margin)),
                        axis=0) - lamb * np.sum(abs(W), axis=0)

    return gradient_b, gradient_w, likelihood


def logistic_train(X, Y, alpha, lamb, max_iterations=10000):
    M, N = X.shape
    W = np.random.randn(N, 1) * 0.01
    bias = 0.0
    iterations = 0
    (G_b, G_w, likelihood) = calculate_gradients(W, X, bias, Y, lamb)
    current_likelihood = likelihood

    while iterations < max_iterations:
        previous_likelihood = current_likelihood
        W = W + alpha * G_w
        bias = bias + alpha * G_b
        iterations += 1
        (G_b, G_w, current_likelihood) = calculate_gradients(W, X, bias, Y, lamb)

        if abs(current_likelihood - previous_likelihood) < 1e-7:
            break

    return W, bias, iterations


def calculate_accuracy(features, labels, weights, bias):
    num_samples, _ = features.shape
    predicted_scores = np.dot(features, weights) + bias
    predicted_labels = np.sign(predicted_scores)

    misclassifications = np.sum(np.abs(labels - predicted_labels)) / 2
    accuracy_percentage = (1 - misclassifications / num_samples) * 100

    return accuracy_percentage


def logistic_validate(X_train, Y_train, X_val, Y_val):
    best_W = best_bias = best_alpha = best_lamb = None
    best_accuracy = 0
    for i in range(-5, -4):
        alpha = 10 ** i
        for j in range(-6, 5):
            lamb = 10 ** j
            (W, bias, iterations) = logistic_train(X_train, Y_train, alpha, lamb)
            current_accuracy = calculate_accuracy(X_val, Y_val, W, bias)

            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                best_W = W
                best_bias = bias
                best_alpha = alpha
                best_lamb = lamb

    return best_W, best_bias, best_alpha, best_lamb


(W, bias, alpha, lamb) = logistic_validate(train_X, train_Y, val_X, val_Y)

test_accuracy = calculate_accuracy(test_X, test_Y, W, bias)
print('Test Accuracy: ', test_accuracy)
print('W: ', W)
print('bias: ', bias)
print('lambda: ', lamb)
