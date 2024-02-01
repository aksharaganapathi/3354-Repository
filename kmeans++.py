import pandas as pd
import numpy as np

data_train = pd.read_csv('leaf.data', header=None).values


def normalize_data(X):
    feature_means = np.mean(X, axis=0)
    feature_stds = np.std(X, axis=0)
    normalized_X = (X - feature_means) / feature_stds
    return normalized_X


class KMeansPlusPlus(object):

    def __init__(self, n_clusters, tolerance, max_iterations):
        self.n_clusters = n_clusters
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.cluster_centers = None
        self.labels = None
        self.inertia = None
        self.initial_cluster_centers = None

    def initialize_cluster_centers(self, X):
        num_samples, num_features = X.shape
        cluster_centers = []
        initial_point = (X[np.random.randint(0, num_samples), :]).reshape(1, -1)
        cluster_centers.append(initial_point)

        for _ in range(self.n_clusters - 1):
            distances = []
            for i in range(num_samples):
                X_i = X[i]
                min_distance = 1e+20
                for center in cluster_centers:
                    current_distance = np.sum((X_i - center) ** 2)
                    if current_distance < min_distance:
                        min_distance = current_distance
                distances.append(min_distance)

            distances = np.array(distances).reshape(1, num_samples)
            distances_sum = np.sum(distances)
            distances = distances / distances_sum

            probabilities = distances.flatten().tolist()
            sampled_choice = np.random.choice(list(range(0, num_samples)), 1, p=probabilities)

            new_cluster_center = X[sampled_choice]
            cluster_centers.append(new_cluster_center)

        return np.array(cluster_centers).reshape(self.n_clusters, num_features)

    def update_cluster_centers(self, X, labels, previous_centers):
        cluster_centers = []

        for i in range(self.n_clusters):
            cluster_i = X[np.where(labels == i)[0]]
            num_samples_cluster_i, num_features_cluster_i = cluster_i.shape
            center_i = previous_centers[i] if num_samples_cluster_i == 0 else np.mean(cluster_i, axis=0)
            cluster_centers.append(center_i)
        return cluster_centers

    def update_labels(self, data, cluster_centers):
        num_samples, num_features = data.shape
        updated_labels = []

        for sample_index in range(num_samples):
            current_sample = data[sample_index]
            best_cluster = None
            min_distance = float('inf')

            for cluster_index in range(self.n_clusters):
                distance_to_cluster = np.dot(cluster_centers[cluster_index] - current_sample,
                                             cluster_centers[cluster_index] - current_sample)

                if distance_to_cluster < min_distance:
                    min_distance = distance_to_cluster
                    best_cluster = cluster_index

            updated_labels.append(best_cluster)

        return np.array(updated_labels).reshape(num_samples, 1)

    def compute_inertia(self, X, labels, cluster_centers):
        inertia = 0
        for i in range(self.n_clusters):
            cluster_i = X[np.where(labels == i)[0]]
            num_samples_cluster_i, num_features_cluster_i = cluster_i.shape
            if num_samples_cluster_i == 0:
                continue
            else:
                inter_cluster_distance = cluster_i - cluster_centers[i]
                inertia = inertia + np.sum(inter_cluster_distance ** 2)

        return inertia

    def fit(self, X):
        X = normalize_data(X)
        cluster_centers = self.initialize_cluster_centers(X)
        labels = self.update_labels(X, cluster_centers)
        inertia = self.compute_inertia(X, labels, cluster_centers)

        iterations = 0
        is_local_optimum = False
        while not is_local_optimum:
            labels = self.update_labels(X, cluster_centers)
            cluster_centers = self.update_cluster_centers(X, labels, cluster_centers)

            current_inertia = self.compute_inertia(X, labels, cluster_centers)

            if abs(inertia - current_inertia) < self.tolerance or iterations > self.max_iterations:
                is_local_optimum = True
                self.labels = labels
                self.cluster_centers = cluster_centers
                self.inertia = inertia

            inertia = current_inertia
            iterations = iterations + 1


M, N = data_train.shape
cluster_centers_column = data_train[:, 0]
X_train = data_train[:, 1:N]

k_values = [12, 18, 24, 36, 42]
tolerance = 1e-17
max_iterations = 1e+4

inertia_matrix = np.empty((20, 5))

for i in range(20):
    for j in range(len(k_values)):
        k = k_values[j]
        kmeans = KMeansPlusPlus(k, tolerance, max_iterations)
        kmeans.fit(X_train)
        inertia_matrix[i][j] = kmeans.inertia

print(inertia_matrix)
