import pandas as pd
import numpy as np
import math

train_data = pd.read_csv('leaf.data', header=None).values
row, col = train_data.shape
X = train_data[:, 1:col]


def normalize(X):
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    return (X - X_mean) / X_std


def multivariate_gaussian(data_point, mean_vector, covariance_matrix):
    dimension = len(mean_vector)

    covariance_determinant = np.linalg.det(covariance_matrix)
    covariance_inverse = np.linalg.inv(covariance_matrix)

    data_minus_mean = (data_point - mean_vector).reshape(1, dimension)
    exponential_term = np.exp(-0.5 * np.dot(np.dot(data_minus_mean, covariance_inverse), data_minus_mean.transpose()))
    normalization_constant = np.sqrt(((2 * math.pi) ** dimension) * covariance_determinant)

    return exponential_term / normalization_constant


def update_cluster_parameters(posteriors, data):
    num_data_points, num_features = data.shape
    num_clusters, _ = posteriors.shape

    updated_means = np.zeros((num_clusters, num_features))
    updated_covariances = np.zeros((num_clusters, num_features, num_features))
    updated_weights = np.zeros(num_clusters)

    for cluster in range(num_clusters):
        cluster_posteriors = posteriors[cluster].reshape(num_data_points, 1)
        total_posterior = np.sum(cluster_posteriors)

        updated_mean = (np.sum(cluster_posteriors * data, axis=0) / total_posterior).reshape(1, num_features)
        updated_means[cluster] = updated_mean

        updated_covariance = np.zeros((num_features, num_features))

        for point in range(num_data_points):
            data_point_diff = (data[point] - updated_mean).reshape(1, num_features)
            updated_covariance += cluster_posteriors[point] * np.outer(data_point_diff, data_point_diff)

        updated_covariance /= total_posterior
        updated_covariance += 1e-6 * np.identity(num_features)
        updated_covariances[cluster] = updated_covariance

        updated_weight = total_posterior / num_data_points
        updated_weights[cluster] = updated_weight

    return updated_means, updated_covariances, updated_weights


class GaussianMixtureModel:
    def __init__(self, num_clusters, convergence_tolerance, max_iterations):
        self.num_clusters = num_clusters
        self.convergence_tolerance = convergence_tolerance
        self.max_iterations = max_iterations
        self.cluster_centers = None
        self.labels = None
        self.inertia = None
        self.initial_cluster_centers = None

    def get_initial_centers(self, X):
        row, col = X.shape
        K = self.num_clusters
        centers = np.random.choice(np.arange(-3, 4, 1), size=(K, col))
        return centers

    def get_initial_parameters(self, X):
        row, col = X.shape
        num_clusters = self.num_clusters
        initial_means = self.get_initial_centers(X)
        covariances = np.tile(np.eye(col), (num_clusters, 1, 1))
        lambdas = np.ones((num_clusters, 1)) / num_clusters
        return initial_means, covariances, lambdas

    def calculate_posteriors(self, cluster_means, cluster_covariances, cluster_weights, data):
        num_clusters = len(cluster_means)
        num_data_points, num_features = data.shape

        probabilities = np.empty((num_clusters, num_data_points))

        for k in range(num_clusters):
            mean_k = cluster_means[k]
            covariance_k = cluster_covariances[k]
            weight_k = cluster_weights[k]

            for m in range(num_data_points):
                data_point = data[m]
                probability = multivariate_gaussian(data_point, mean_k, covariance_k)
                probabilities[k][m] = weight_k * probability

        cumulative_probabilities = np.sum(probabilities, axis=0)
        posteriors = probabilities / cumulative_probabilities

        return posteriors

    def compute_log_likelihood(self, cluster_means, cluster_covariances, cluster_weights, data):
        num_data_points, num_features = data.shape
        num_clusters = self.num_clusters
        total_log_likelihood = 0

        for point in range(num_data_points):
            point_likelihood = 0

            for cluster in range(num_clusters):
                point_likelihood += cluster_weights[cluster] * multivariate_gaussian(data[point],
                                                                                     cluster_means[cluster],
                                                                                     cluster_covariances[cluster])

            total_log_likelihood += np.log(point_likelihood)

        return total_log_likelihood

    def fit(self, X):
        means, covariances, lambdas = self.get_initial_parameters(X)
        previous_log_likelihood = 0
        current_log_likelihood = 1e+10
        iterations = 0
        posteriors = 0

        while abs(current_log_likelihood - previous_log_likelihood) > self.convergence_tolerance:
            posteriors = self.calculate_posteriors(means, covariances, lambdas, X)
            means, covariances, lambdas = update_cluster_parameters(posteriors, X)

            previous_log_likelihood = current_log_likelihood
            current_log_likelihood = self.compute_log_likelihood(means, covariances, lambdas, X)
            iterations += 1

        return means, covariances, lambdas, current_log_likelihood, posteriors


k_list = [12, 18, 24, 36, 42]
tolerance = 1e-4
max_iterations = 1e+4

log_likelihood_matrix = np.empty((20, 5))

for i in range(20):
    for j, k in enumerate(k_list):
        gmm = GaussianMixtureModel(k, tolerance, max_iterations)
        means, covariances, lambdas, current_log_likelihood, posteriors = gmm.fit(X)
        log_likelihood_matrix[i][j] = current_log_likelihood

converged_log_likelihoods = log_likelihood_matrix[:, -1]

mean_values = np.mean(converged_log_likelihoods)
variance_values = np.var(converged_log_likelihoods)

for j, k in enumerate(k_list):
    print(f"Mean log-likelihood for k={k}: {mean_values[j]}")
    print(f"Variance log-likelihood for k={k}: {variance_values[j]}")
