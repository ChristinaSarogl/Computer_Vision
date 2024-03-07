import numpy as np
import matplotlib.pyplot as plt
import cv2

def kmeans(data, K, thresh, n_iter, n_attempts):
    """
    Cluster data in K clusters using the K-Means algorithm
    :param data: numpy.array(float), the input data array with N (#data) x D (#feature_dimensions) dimensions
    :param K: int, number of clusters
    :param thresh: float, convergence threshold
    :param n_iter: int, #iterations of the K-Means algorithm
    :param n_attempts: int, #attempts to run the K-Means algorithm
    :return:
    compactness: float, the sum of squared distance from each point to their corresponding centers
    labels: numpy.array(int), the label array with Nx1 dimensions, where it denotes the corresponding cluster of
    each data point
    centers : numpy.array(float), a KxD array with the final centroids
    """

    #Sanity checks
    assert len(data.shape) == 2
    assert K > 0 and K <= data.shape[0]
    assert thresh > 0
    assert n_iter > 0
    assert n_attempts > 0

    attemp_comp = []
    attemp_labels =[]
    attemp_centres =[]
    tested_idx = []

    for a in range(n_attempts):
        idx = np.random.choice(data.shape[0], K, replace=False)

        #Check that the generated centroid have not been investigated before
        if (len(tested_idx) != 0):
            while(True):
                duplicates = np.array([index in tested_idx for index in idx])
                dupl_idx = np.where(duplicates == True)[0]

                if (dupl_idx.shape[0] != 0):
                    new_idx = np.random.choice(data.shape[0], dupl_idx.shape[0], replace=False)
                    idx = np.delete(idx, dupl_idx)
                    idx = np.append(idx, new_idx)
                else:
                    tested_idx = np.append(tested_idx, idx)
                    break
        else:
            tested_idx = idx

        centroids = data[idx]

        for i in range(n_iter):
            #Assign points to centroids
            distances = np.array([np.sum((data-centroid)**2, axis=1) for centroid in centroids])
            labels = np.argmin(distances,axis=0)

            #Update centroids
            new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(K)])

            # Break if centroids have converged
            covergence = np.sum(np.abs(new_centroids-centroids))
            if (covergence <= thresh):
                break

            centroids = new_centroids

        # Calculate compactness
        clusters = [data[labels == i] for i in range(K)]
        cluster_dist = [np.sum(np.sum((clusters[c] - centroids[c]) ** 2, axis=1)) for c in
                        range(len(clusters))]

        attemp_comp.append(np.sum(cluster_dist))
        attemp_labels.append(labels)
        attemp_centres.append(centroids)

    min_index = attemp_comp.index(min(attemp_comp))
    return attemp_comp[min_index], attemp_labels[min_index], attemp_centres[min_index]


def plot_images(title,result, result_cv2):
    plt.figure(title)
    plt.subplot(1, 2, 1)
    plt.imshow(result)
    plt.axis("off")
    plt.title("Mine")

    plt.subplot(1, 2, 2)
    plt.imshow(result_cv2)
    plt.axis("off")
    plt.title("OpenCV")