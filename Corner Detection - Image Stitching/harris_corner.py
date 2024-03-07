import numpy as np
import cv2
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

def plot_corners(imgs, title,**kwargs):
    img_corners = kwargs.get("img_corners", None)
    plt.figure(title)

    for i in range(1, len(imgs)+1):
        plt.subplot(2,3,i)
        if img_corners != None:
            results = imgs[i - 1].copy()
            for corner in img_corners[i-1]:
                x, y = corner.astype(np.int32).ravel()
                cv2.circle(img=results, center=(x, y), radius=4, color=[255,0,0], thickness=-1,
                           lineType=cv2.LINE_AA)
            plt.imshow(cv2.cvtColor(results, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(cv2.cvtColor(imgs[i - 1], cv2.COLOR_BGR2RGB))
        plt.axis("off")

def detect_corners(input_image, max_corners=0, quality_level=0.01, min_distance=10, block_size=5, k=0.05):
    """
    Detect corners using Harris Corner Detector
    :param input_image: numpy.array(uint8 or float), input 8-bit or foating-point 32-bit, single-channel
    image
    :param max_corners: int, maximum number of corners to return, if 0 then return all
    :param quality_level: float, parameter characterizing the minimal accepted quality of image corners
    :param min_distance: float, minimum possible Euclidean distance between the returned corners
    :param block_size: int, size of an average block for computing a derivative covariation matrix
    over each pixel neighborhood.
    :param k: float, free parameter of the Harris detector
    :return:
    corners: numpy.array(uint8)), corner coordinates for each input image
    """

    #Sanity checks
    assert max_corners >= 0
    assert quality_level >= 0 and quality_level <= 1
    assert min_distance >= 0
    assert block_size%2 != 0 and block_size > 0
    assert k >= 0.04 and k <= 0.06

    # Calculate partial derivative for x
    der_x = cv2.Sobel(input_image, -1, dx=1, dy=0, borderType=cv2.BORDER_REPLICATE)
    # Calculate partial derivative for y
    der_y = cv2.Sobel(input_image, -1, dx=0, dy=1, borderType=cv2.BORDER_REPLICATE)

    indx_R_scores = []

    # Add padding to the arrays
    pad_size = block_size // 2
    der_x_padd = cv2.copyMakeBorder(src=der_x, top=pad_size, bottom=pad_size, left=pad_size, right=pad_size,
                                    borderType=cv2.BORDER_CONSTANT, value=0)
    der_y_padd = cv2.copyMakeBorder(src=der_y, top=pad_size, bottom=pad_size, left=pad_size, right=pad_size,
                                    borderType=cv2.BORDER_CONSTANT, value=0)

    for r in range(pad_size, der_x.shape[0] + pad_size):
        for c in range(pad_size, der_x.shape[1] + pad_size):
            #Find the neighbours
            x = der_x_padd[r - pad_size:r + pad_size + 1, c - pad_size:c + pad_size + 1]
            y = der_y_padd[r - pad_size:r + pad_size + 1, c - pad_size:c + pad_size + 1]

            #Calculate the structure matrix
            Sum_IxIx = np.sum(np.multiply(x, x))
            Sum_IyIy = np.sum(np.multiply(y, y))
            Sum_IxIy = np.sum(np.multiply(x, y))

            #Calculate the R-score
            det_pix = (Sum_IxIx * Sum_IyIy) - (Sum_IxIy ** 2)
            trace_pix = (Sum_IxIx + Sum_IyIy) ** 2
            R_score = det_pix - k * trace_pix
            indx_R_scores.append([r - pad_size, c - pad_size, R_score])

    #Convert list to np.array
    indx_R_scores = np.array(indx_R_scores, dtype= np.single)

    #Find minimum quality
    ratio = indx_R_scores[:,2].max() * np.float32(quality_level)
    pot_corners = np.array([pix for pix in indx_R_scores if pix[2] >= ratio])

    #Sort in descenting order
    pot_corners = pot_corners[pot_corners[:, 2].argsort()[::-1]]

    # Create KD-tree
    points_kdtree = KDTree(pot_corners[:,:2])
    corners = []
    ignore_list = []

    for corner_indx in range(pot_corners.shape[0]):
        if corner_indx in ignore_list:
            continue

        # Get indexes of neighbours
        neighbours_indexes = points_kdtree.query_ball_point(pot_corners[corner_indx][:2], min_distance)

        # Remove self from neighbours
        neighbours_indexes.remove(corner_indx)
        # Add corner to corner list (set width first, then height)
        corners.append([pot_corners[corner_indx][1],pot_corners[corner_indx][0]])

        #Add neighbours to the ignore list
        for entry in neighbours_indexes:
            if entry not in ignore_list:
                ignore_list.append(entry)

    corners = np.array(corners)

    #Return cornerns based on the max_corners value
    if max_corners == 0 or max_corners > corners.shape[0]:
        return corners[:,:2]
    else:
        return corners[:max_corners,:2]