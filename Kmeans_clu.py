import numpy as np
import matplotlib.pyplot as plt

## reading the file
fyle = open("data.csv")
lines = fyle.readlines()
fyle.close()

lyst = []
for line in lines:
    x_val, y_val = line.split(",")
    lyst.append([float(x_val), float(y_val)])
data = np.array(lyst)

## visualization
plt.figure(facecolor="black", figsize=(6, 6))
plt.scatter(data[:,0], data[:,1], marker=".", s=0.5, c="white")
plt.gca().set_facecolor("black") 
plt.xticks(color="white")
plt.yticks(color="white")
plt.show()

## define a function to compute dist
def _compute_dist(data, centroids):
    dist_matrix = np.zeros(shape=(data.shape[0], centroids.shape[0]))
    for p_idx, p in enumerate(data):
        for c_idx, c in enumerate(centroids):
            dist_matrix[p_idx][c_idx] = np.sqrt((p[0]-c[0])**2 + (p[1]-c[1])**2)
    return dist_matrix

def compute_silh_score(clu_labels, centroids):
    for centeroid in centroids:
        dists = np.sum((centroids - centeroid) ** 2, axis=1)
        cl_clu_id = np.argsort(dists)[1]

    silh_scores = []
    for sample_idx, sample in enumerate(data):
        clu_id = clu_labels[sample_idx]
        same_clu = data[clu_labels == clu_id]
        closest_clu = data[clu_labels == cl_clu_id]
        a = np.sqrt(np.sum((sample - same_clu) ** 2, axis=1)).mean()
        b = np.sqrt(np.sum((sample - closest_clu) ** 2, axis=1)).mean()

        silh_score = (b - a) / max(a, b)
        silh_scores.append(silh_score)
    
    return np.mean(silh_scores)

## kmeans clustering function
def kmeans_clu(data, k, n_iteration, random_state=0):

    ## initiate centroids
    np.random.seed(random_state)
    x_min, x_max = data[:,0].min(), data[:,0].max()
    y_min, y_max = data[:,1].min(), data[:,1].max()
    centroids = np.array([np.random.uniform(x_min, x_max, k), 
                            np.random.uniform(y_min, y_max, k)]).T
    
    plt.figure(facecolor="black", figsize=(6, 6))
    plt.gca().set_facecolor("black") 
    plt.xticks(color="white")
    plt.yticks(color="white")

    for _ in range(n_iteration):
        # compute distances and assign clusters
        distances = _compute_dist(data, centroids)
        cluster_labels = np.argmin(distances, axis=1)

        # update centroids
        new_centroids = []
        for i in range(k):
            new_centroid = data[cluster_labels == i].mean(axis=0)
            new_centroids.append(new_centroid)
        centroids = np.array(new_centroids)

        ## plotting
        plt.scatter(data[:,0], data[:,1], c=cluster_labels, cmap=None, s=0.5)
        plt.draw() 
        plt.pause(0.3)
    
    return cluster_labels, centroids

## run clustering
for k in range(2, 9):
    clu_labels, centroids = kmeans_clu(data, k=k, n_iteration=15, random_state=0)
    silh_score = compute_silh_score(clu_labels, centroids)
    print(f"{k}: {round(silh_score, 3)}")
