import sklearn
import numpy as np

def cosine_similarity(x, y):
    if y.shape[0] != 1:
        sim = [np.dot(x, y[j]) / (np.linalg.norm(x) * np.linalg.norm(y[j])) for j in range(y.shape[0])]
        return sum(sim) / len(sim)
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def euclidean_distance(x, y):
    if y.shape[0] != 1:
        dis = [np.linalg.norm(x - y[j]) for j in range(y.shape[0])]
        return sum(dis) / len(dis)
    return np.linalg.norm(x - y)

def distance_based(measure="cosine"):
    def distance_attack(collect_data, known_data, known_label):
        if measure == "cosine":
            similarity = cosine_similarity
        else:
            similarity = euclidean_distance
        collect_label = []
        for i in range(collect_data.shape[0]):
            sim = [similarity(collect_data[i], known_data[j]) for j in range(known_data.shape[0])]
            collect_label.append(known_label[np.argmax(sim)])
        return np.array(collect_label)
    return distance_attack

def kmeans_based(collect_data, known_data, known_label):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=len(np.unique(known_label)), random_state=0).fit(collect_data)
    collect_label = kmeans.predict(known_data)
    pred_idx = kmeans.predict(collect_data)
    pred_label = [known_label[collect_label[i]] for i in pred_idx]
    return np.array(pred_label)

def random_known_data(known_data, known_label, num):
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