import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.neighbors import NearestNeighbors

def plot_image(title, img):
    plt.figure(title)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")

def nn_graph(input_image, k):
    """
    Create a graph based on the k-nearest neighbors of each pixel in the (i,j,r,g,b) feature space.
    Edge weights are calculated as the Euclidean distance of the node's features
    and its corresponding neighbors.
    :param input_image: numpy.array(uint8), input image of HxWx3 dimensions
    :param k: int, nearest neighbors for each node
    :return:
    graph: tuple(V: numpy.array(int), E: <graph connectivity representation>), the NN-graph where
    V is the set of pixel-nodes of (W*H)x2 dimensions and E is a representation of the graph's
    undirected edges along with their corresponding weight
    """

    #Sanity checks
    assert len(input_image.shape) == 3
    assert k > 0

    print("Creating Graph")

    #Create feature space (i, j, r, g, b)
    coord_val = np.zeros((input_image.shape[0], input_image.shape[1], 5), dtype=np.float32)

    for r in range(input_image.shape[0]):
        for c in range(input_image.shape[1]):
            coord_val[r][c] = np.hstack((r, c, input_image[r, c]))

    #Reshape points into a 2D array
    coord_val = np.reshape(coord_val, (coord_val.shape[0] * coord_val.shape[1], 5)).astype(np.float32)

    nbrs = NearestNeighbors().fit(coord_val)
    adj_matrix = nbrs.kneighbors_graph(coord_val, n_neighbors=k+1, mode='distance')

    graph = (coord_val[:, :2], adj_matrix)
    return graph

def segmentation(G, k, min_size):
    """
    Segment the image base on the Efficient Graph-Based Image Segmentation algorithm.
    :param G: tuple(V, E), the input graph
    :param k: int, sets the threshold k/|C|
    :param min_size: int, minimum size of clusters
    :return:
    clusters: numpy.array(int), a |V|x1 array where it denotes the cluster for each node v of the graph
    """
    #Sanity checks
    assert k > 0
    assert min_size > 0

    print("Segmenting Image")
    V, E = G
    pixel_num = V.shape[0]

    neighbours_per_px = np.split(E.indices, pixel_num)
    weights_per_px_neighbour = np.split(E.data, pixel_num)

    #Remove first column because the pixel is connected to itself
    neighbours_per_px = np.delete(neighbours_per_px, 0, axis=1)
    weights_per_px_neighbour = np.delete(weights_per_px_neighbour, 0, axis=1)

    #Sort edge weights
    flatten_weights = np.reshape(weights_per_px_neighbour, (weights_per_px_neighbour.shape[0] * weights_per_px_neighbour.shape[1]))
    sort_idx = np.argsort(flatten_weights)

    #Array storing the clusters
    clusters = [np.array([i]) for i in range(pixel_num)]

    #Array storing the cluster of each pixel
    clusters_idx = np.arange(pixel_num)

    #Array storing the weights of each cluster
    cluster_weights = [c for c in np.zeros(pixel_num).reshape(pixel_num,1)]

    #Segment image
    for q in range(sort_idx.shape[0]):
        weight_idx = sort_idx[q]

        # Find vertices on edge Eq
        Vi = weight_idx // neighbours_per_px.shape[1] #row index in weights graph
        Vj = weight_idx % neighbours_per_px.shape[1] #col index in weights graph

        pixel_i = Vi
        pixel_j = neighbours_per_px[Vi][Vj]

        cluster_idx_i = clusters_idx[pixel_i]
        cluster_idx_j = clusters_idx[pixel_j]

        if (cluster_idx_i != cluster_idx_j):
            Ci = clusters[cluster_idx_i]
            Cj = clusters[cluster_idx_j]

            weights_i = cluster_weights[cluster_idx_i]
            weights_j = cluster_weights[cluster_idx_j]

            min_internal = get_min_internal(Ci, Cj, k, weights_i, weights_j)

            if (flatten_weights[weight_idx] <= min_internal):
                #Merge the clusters
                clusters[cluster_idx_i] = np.append(clusters[cluster_idx_i], Cj)

                #Add the weight of the new edge
                if ((weights_i.shape[0] == 1) and (weights_i == 0)):
                    if ((weights_j.shape[0] == 1) and (weights_j == 0)):
                        cluster_weights[cluster_idx_i] = np.array([flatten_weights[weight_idx]])
                    else :
                        #Add previous cluster weights to the new cluster
                        cluster_weights[cluster_idx_i] = weights_j
                        cluster_weights[cluster_idx_i] = np.append(cluster_weights[cluster_idx_i],
                                                                   flatten_weights[weight_idx])
                else:
                    if ((weights_j.shape[0] == 1) and (weights_j == 0)):
                        cluster_weights[cluster_idx_i] = np.append(cluster_weights[cluster_idx_i], flatten_weights[weight_idx])
                    else:
                        # Add previous cluster weights to the new cluster
                        cluster_weights[cluster_idx_i] = np.append(weights_i, weights_j)
                        cluster_weights[cluster_idx_i] = np.append(cluster_weights[cluster_idx_i],
                                                                   flatten_weights[weight_idx])

                        #Delete merged cluster and respective weight
                del clusters[cluster_idx_j]
                del cluster_weights[cluster_idx_j]

                #Update pixel cluster index
                clusters_idx[Cj] = cluster_idx_i
                clusters_idx[clusters_idx > cluster_idx_j] -= 1

    #Post processing
    new_clusters = clusters.copy()

    for c_idx in range(len(clusters)):
        cluster = clusters[c_idx]
        if (cluster.shape[0] < min_size):
            new_c_indx = clusters_idx[cluster[0]]
            neigh_clusters = []

            for pixel in cluster:
                cl = clusters_idx[neighbours_per_px[pixel]]
                [neigh_clusters.append(c) for c in cl]

            unique, counts = np.unique(np.array(neigh_clusters), return_counts=True)

            # Check if the cluster has only inner connections
            closest_cl_id = new_c_indx
            closest_connections = np.argsort(counts)[::-1]
            for connection in closest_connections:
                if (unique[connection] != new_c_indx):
                    closest_cl_id = unique[connection]
                    break

            if (closest_cl_id != new_c_indx):
                # Connect the clusters
                new_clusters[closest_cl_id] = np.append(new_clusters[closest_cl_id], cluster)

                # Delete merged cluster
                del new_clusters[new_c_indx]

                # Update pixel cluster index
                clusters_idx[cluster] = closest_cl_id
                clusters_idx[clusters_idx > new_c_indx] -= 1

    return clusters_idx


def get_min_internal(cluster_i, cluster_j, k, weights_i, weights_j):
    internal_i = weights_i.max()
    internal_j = weights_j.max()

    thres_i = k / cluster_i.shape[0]
    thres_j = k / cluster_j.shape[0]

    return min(internal_i + thres_i, internal_j + thres_j)
