import pandas as pd
import numpy as np
import math

train_data = pd.read_csv('leaf.data', header=None).values


def gaussian_probability(data_point, mean, covariance):
    num_features = len(mean)
    covariance_determinant = np.linalg.det(covariance)
    covariance_inverse = np.linalg.inv(covariance)

    X_mean = (data_point - mean).reshape(1, num_features)
    exponential = np.exp(-0.5 * np.dot(np.dot(X_mean, covariance_inverse), X_mean.transpose()))
    constant = np.sqrt(((2 * math.pi) ** num_features) * covariance_determinant)
    return exponential / constant


def maximization_step(posteriors, data):
    num_data_points, num_features = data.shape
    num_clusters, num_data_points = posteriors.shape

    cluster_means = np.zeros((num_clusters, num_features))
    cluster_covariances = np.zeros((num_clusters, num_features, num_features))
    cluster_weights = np.zeros(num_clusters)

    for cluster_index in range(num_clusters):
        posterior_probabilities = posteriors[cluster_index].reshape(num_data_points, 1)
        posterior_sum = np.sum(posterior_probabilities)

        mean_cluster = (np.sum(posterior_probabilities * data, axis=0) / posterior_sum).reshape(1, num_features)
        cluster_means[cluster_index] = mean_cluster

        covariance_cluster = np.zeros((num_features, num_features))

        for data_point_index in range(num_data_points):
            data_point_minus_mean = (data[data_point_index] - mean_cluster).reshape(1, num_features)
            covariance_cluster = covariance_cluster + posterior_probabilities[data_point_index] * np.outer(data_point_minus_mean, data_point_minus_mean)

        covariance_cluster /= posterior_sum
        covariance_cluster += 1e-6 * np.identity(num_features)
        cluster_covariances[cluster_index] = covariance_cluster

        weight_cluster = posterior_sum / num_data_points
        cluster_weights[cluster_index] = weight_cluster

    return cluster_means, cluster_covariances, cluster_weights


class GaussianMixtureModel(object):

    def __init__(self, num_clusters, tolerance, max_iterations):
        self.num_clusters = num_clusters
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.cluster_centers = None
        self.labels = None
        self.inertia = None
        self.initial_cluster_centers = None

    def normalize_data(self, data):
        data_mean = np.mean(data, axis=0)
        data_std = np.std(data, axis=0)
        return (data - data_mean) / data_std

    def get_initial_centers(self, data):
        num_data_points, num_features = data.shape
        num_clusters = self.num_clusters

        cluster_centers = []
        initial_point = data[np.random.choice(num_data_points)].reshape(1, num_features)
        cluster_centers.append(initial_point)

        for i in range(num_clusters - 1):
            distances = []
            for j in range(num_data_points):
                data_j = data[j]
                min_distance = 10000000
                for c in range(len(cluster_centers)):
                    data_j_c = data_j - cluster_centers[c]
                    current_distance = np.inner(data_j_c, data_j_c)
                    if current_distance < min_distance:
                        min_distance = current_distance
                distances.append(min_distance)

            distances = np.array(distances).reshape(1, num_data_points)
            distances_sum = np.sum(distances)
            distances = distances / distances_sum

            probabilities = distances.flatten().tolist()
            choice = np.random.choice(list(range(0, num_data_points)), 1, p=probabilities)

            new_cluster_center = data[choice]
            cluster_centers.append(new_cluster_center)

        return np.array(cluster_centers).reshape(num_clusters, num_features)

    def get_initial_parameters(self, data):
        num_data_points, num_features = data.shape
        num_clusters = self.num_clusters
        means = self.get_initial_centers(data)
        covariances = np.empty((num_clusters, num_features, num_features))

        for k in range(num_clusters):
            covariances[k] = np.eye(num_features)

        weights = np.ones((num_clusters, 1)) * (1 / num_clusters)
        return means, covariances, weights

    def expectation_step(self, means, covariances, weights, data):
        num_clusters = self.num_clusters
        num_data_points, num_features = data.shape
        probabilities = np.empty((num_clusters, num_data_points))

        for k in range(num_clusters):
            mean_k = means[k]
            covariance_k = covariances[k]
            weight_k = weights[k]

            for m in range(num_data_points):
                data_m = data[m]
                probability = gaussian_probability(data_m, mean_k, covariance_k)
                probabilities[k][m] = weight_k * probability

        cumulative_probabilities = np.sum(probabilities, axis=0)
        posteriors = probabilities / cumulative_probabilities

        return posteriors

    def compute_log_likelihood(self, means, covariances, weights, data):
        num_data_points, num_features = data.shape
        num_clusters = self.num_clusters
        log_likelihood = 0

        for m in range(num_data_points):
            data_m_prob = 0
            for k in range(num_clusters):
                data_m_prob = data_m_prob + weights[k] * gaussian_probability(data[m], means[k], covariances[k])
            log_likelihood = log_likelihood + np.log(data_m_prob)

        return log_likelihood

    def fit(self, data):
        (means, covariances, weights) = self.get_initial_parameters(data)
        previous_log_likelihood = 0
        current_log_likelihood = 1e+10
        iterations = 0

        while abs(current_log_likelihood - previous_log_likelihood) > self.tolerance:
            posteriors = self.expectation_step(means, covariances, weights, data)
            (means, covariances, weights) = maximization_step(posteriors, data)

            previous_log_likelihood = current_log_likelihood
            current_log_likelihood = self.compute_log_likelihood(means, covariances, weights, data)
            iterations = iterations + 1

        return means, covariances, weights, current_log_likelihood


num_data_points, num_features = train_data.shape
cluster_centers_column = train_data[:, 0]
data_matrix = train_data[:, 1:num_features]

num_clusters_list = [12, 18, 24, 36, 42]
tolerance_value = 1e-4
max_iterations_value = 1e+4

log_likelihood_matrix = np.empty((20, 5))

for i in range(20):
    for j in range(len(num_clusters_list)):
        num_clusters = num_clusters_list[j]
        gmm = GaussianMixtureModel(num_clusters, tolerance_value, max_iterations_value)
        (means, covariances, weights, current_log_likelihood) = gmm.fit(data_matrix)
        log_likelihood_matrix[i][j] = current_log_likelihood
