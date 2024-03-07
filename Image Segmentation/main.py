import numpy as np
import matplotlib.pyplot as plt
import cv2

import k_means
import image_segmentation as imseg

def kmeans():
    print("K-Means")
    img = cv2.imread('data/home.jpg', cv2.IMREAD_UNCHANGED)
    #Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    rgb_vectors = np.reshape(img, (img.shape[0] * img.shape[1], 3)).astype(np.float32)

    #Pixel coordinates and rgb values
    coord_val = np.zeros((img.shape[0],img.shape[1],5))

    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            coord_val[r][c] = np.hstack((r, c, img[r,c]))

    coord_val = np.reshape(coord_val, (coord_val.shape[0] * coord_val.shape[1], 5)).astype(np.float32)

    K, thresh, n_iter, n_attempts = 4, 1.0, 10, 10
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, n_iter, thresh)

    print("Generating clusters")
    compactness, label, center = k_means.kmeans(rgb_vectors, K, thresh, n_iter, n_attempts)
    compactness_cv2, label_cv2, center_cv2 = cv2.kmeans(rgb_vectors, K, None, criteria, n_attempts,
                                                        cv2.KMEANS_RANDOM_CENTERS)

    compactness_coord, label_coord, center_coord = k_means.kmeans(coord_val, K, thresh, n_iter, n_attempts)
    compactness_coord_cv2, label_coord_cv2, center_coord_cv2 = cv2.kmeans(coord_val, K, None, criteria, n_attempts,
                                                                          cv2.KMEANS_RANDOM_CENTERS)

    print("Generated clusters")
    # Remove coordinates from the centroids
    center_coord = center_coord[:, 2:]
    center_coord_cv2 = center_coord_cv2[:, 2:]

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)

    center_cv2 = np.uint8(center_cv2)
    result_cv2 = center_cv2[label_cv2.flatten()]
    result_cv2 = result_cv2.reshape(img.shape)

    # Pixel coordinates and rgb values
    center_coord = np.uint8(center_coord)
    result_coord = center_coord[label_coord.flatten()]
    result_coord = result_coord.reshape(img.shape)

    center_coord_cv2 = np.uint8(center_coord_cv2)
    result_coord_cv2 = center_coord_cv2[label_coord_cv2.flatten()]
    result_coord_cv2 = result_coord_cv2.reshape(img.shape)

    # Plot results
    k_means.plot_images("RBG values", result, result_cv2)
    k_means.plot_images("Pixel Coordinates & RBG values", result_coord, result_coord_cv2)

    plt.show()

def egbis():
    img = cv2.imread('data/eiffel_tower.jpg', cv2.IMREAD_UNCHANGED)
    imseg.plot_image("Image", img)

    #Blur image
    imgBlur = cv2.GaussianBlur(src=img, ksize=(3, 3), sigmaX=0.8, borderType=cv2.BORDER_REPLICATE)

    graph = imseg.nn_graph(imgBlur, 10)

    k, min_size = 550, 300
    clusters = imseg.segmentation(graph, k, min_size)

    print("Image Segmented")

    ind_clusters = [clusters[clusters == i] for i in range(clusters.max()+1)]
    colors = [int(c * (180 / len(ind_clusters))) for c in range(1,len(ind_clusters) + 1)]

    hsv_values = np.full((clusters.shape[0],3), 255, dtype=np.uint8)
    hsv_values[...,0] = np.array([colors[c] for c in clusters])
    hsv_values = hsv_values.reshape((img.shape[0], img.shape[1], 3))

    seg_img = cv2.cvtColor(hsv_values, cv2.COLOR_HSV2BGR)
    imseg.plot_image("Clusters", seg_img)

    plt.show()

if __name__ == "__main__":
    kmeans()
    egbis()