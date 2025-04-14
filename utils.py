import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import sklearn.preprocessing as preprocessing

def cosine_distance(x: np.array, y: np.array):
    return  - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def euclidean_distance(x: np.array, y: np.array):
    return np.linalg.norm(x - y)

def distance_based(measure="cosine"):
    def distance_attack(collect_data: np.array, known_data: np.array, known_label: dict):
        collect_data = preprocessing.normalize(collect_data)
        known_data = preprocessing.normalize(known_data)
        if measure == "cosine":
            similarity = cosine_distance
        else:
            similarity = euclidean_distance
        collect_label = []
        for i in range(collect_data.shape[0]):
            sim = [similarity(collect_data[i], known_data[j]) for j in range(known_data.shape[0])]
            collect_label.append(known_label[np.argmin(sim)])
        return np.array(collect_label)
    return distance_attack

def kmeans_based(collect_data: np.array, known_data: np.array, known_labels: dict):

    unique_labels = np.unique(known_labels)
    n_clusters = len(unique_labels)
    
    known_dict = {}
    for label in unique_labels:
        known_dict[label] = known_data[known_labels == label]
    
    init_centroids = []
    for label in unique_labels:
        centroid = known_dict[label][np.random.choice(len(known_dict[label]))]
        init_centroids.append(centroid)

    kmeans = KMeans(n_clusters=n_clusters, init=np.array(init_centroids), n_init=1)
    kmeans.fit(collect_data)

    cluster_labels = kmeans.labels_
    known_cluster_labels = kmeans.predict(known_data)
    confusion_matrix = np.zeros((n_clusters, n_clusters), dtype=np.int64)
    for i in range(len(known_labels)):
        true_label_idx = np.where(unique_labels == known_labels[i])[0][0]
        pred_label_idx = known_cluster_labels[i]
        confusion_matrix[true_label_idx, pred_label_idx] += 1
    row_ind, col_ind = linear_sum_assignment(confusion_matrix.max() - confusion_matrix)
    cluster_to_label_map = {col_ind[i]: unique_labels[row_ind[i]] for i in range(len(row_ind))}
    pred_labels = np.array([cluster_to_label_map.get(label, -1) for label in cluster_labels])
    return pred_labels

def random_known_data(known_data, known_label, num=2):
    known_data_list = []
    known_label_list = []
    for i in np.unique(known_label):
        idx = np.where(known_label == i)[0]
        idx = np.random.choice(idx, num, replace=False)
        known_data_list.append(known_data[idx])
        known_label_list.append(known_label[idx])
    known_data = np.concatenate(known_data_list, axis=0)
    known_label = np.concatenate(known_label_list, axis=0)
    return known_data, known_label

# if __name__ == "__main__":
#     data = np.load("test.npy")
#     labels = np.load("test_l.npy")
#     data = preprocessing.normalize(data)

#     known_data, known_label = random_known_data(data, labels, 40)
#     func = kmeans_based
#     res = func(data, known_data, known_label)
#     print(sum(res == labels))