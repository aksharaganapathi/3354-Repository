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


def normalize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_norm = (X - mean) / std
    return X_norm, mean, std


def rescale(X, X_mean, X_std):
    return (X - X_mean) / X_std


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def calculate_class_probabilities(weights, bias, input_data):
    functional_margin = np.dot(input_data, weights) + bias
    probability_positive_class = sigmoid(functional_margin)
    probability_negative_class = 1 - probability_positive_class
    return probability_positive_class, probability_negative_class, functional_margin


def compute_gradients(weights, input_data, bias, labels):
    probability_pos, probability_neg, functional_margin = calculate_class_probabilities(weights, bias, input_data)
    classification_error = (0.5 * (labels + 1) - probability_pos)

    gradient_bias = np.sum(classification_error)
    gradient_weights = np.sum(input_data * classification_error, axis=0).reshape(-1, 1)

    likelihood = np.sum(0.5 * (labels + 1) * functional_margin - np.log(1 + np.exp(functional_margin)))

    return gradient_bias, gradient_weights, likelihood


def logistic_train(X, Y, alpha):
    rows, cols = X.shape
    W = np.ones(cols).reshape(-1, 1) * 0.25
    bias = 0.75
    iterations = 0

    gradient_bias, gradient_w, likelihood = compute_gradients(W, X, bias, Y)
    current_likelihood = likelihood

    while True:
        step = alpha
        previous_likelihood = current_likelihood
        W = W + step * gradient_w
        bias = bias + step * gradient_bias
        iterations += 1
        gradient_bias, gradient_w, current_likelihood = compute_gradients(W, X, bias, Y)

        if abs(current_likelihood - previous_likelihood) < 1e-7:
            break

    return W, bias, iterations


def calculate_accuracy(features, labels, weights, bias):
    num_samples = len(labels)
    predicted_scores = np.dot(features, weights) + bias
    predicted_labels = np.sign(predicted_scores)

    misclassifications = np.sum(np.abs(labels - predicted_labels)) / 2
    accuracy_percentage = (1 - misclassifications / num_samples) * 100

    return accuracy_percentage


def logistic_validate(train_X, train_Y, val_X, val_Y):
    best_W = best_bias = best_alpha = None
    best_accuracy = 0

    for i in range(-5, -3):
        alpha = 10 ** i
        W, bias, iterations = logistic_train(train_X, train_Y, alpha)
        current_accuracy = calculate_accuracy(val_X, val_Y, W, bias)

        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_W, best_bias, best_alpha = W, bias, alpha

    return best_W, best_bias, best_alpha


train_X, X_mean, X_std = normalize(train_X)
val_X = rescale(val_X, X_mean, X_std)
test_X = rescale(test_X, X_mean, X_std)

W, bias, alpha = logistic_validate(train_X, train_Y, val_X, val_Y)

training_accuracy = calculate_accuracy(train_X, train_Y, W, bias)
print('Training Accuracy: ', training_accuracy)

test_accuracy = calculate_accuracy(test_X, test_Y, W, bias)
print('Test Accuracy: ', test_accuracy)