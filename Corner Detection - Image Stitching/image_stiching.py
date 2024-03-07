import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_images_1_5(imgs, title, **kwargs):
    keypoints = kwargs.get("keypoints", None)

    plt.figure(title)

    for i in range(1, len(imgs) + 1):
        plt.subplot(1, 5, i)
        if (keypoints != None):
            img_keypoints = np.empty(imgs[i-1].shape, dtype=np.uint8)
            cv2.drawKeypoints(imgs[i - 1], keypoints[i - 1], img_keypoints)
            plt.imshow(cv2.cvtColor(img_keypoints, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(cv2.cvtColor(imgs[i-1], cv2.COLOR_BGR2RGB))

        plt.axis("off")

def plot_correspond(imgs, title, kp, **kwargs):
    matches = kwargs.get("matches", None)

    plt.figure(title)

    for i in range(1, len(imgs)):
        plt.subplot(2,2,i)
        plt.title("Images {}-{}".format(i, i+1))

        img_matches = np.empty((max(imgs[i-1].shape[0], imgs[i].shape[0]), imgs[i-1].shape[1] + imgs[i].shape[1], 3),
                               dtype=np.uint8)
        cv2.drawMatchesKnn(imgs[i-1], kp[i-1], imgs[i], kp[i], outImg=img_matches, matches1to2=matches[i-1], flags=2)
        plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
        plt.axis("off")

def plot_no_outliers(imgs, title, good_matches, mask, kp):
    inliers = []
    for i in range(len(imgs) - 1):
        good = np.array(good_matches[i])
        inliers.append(good[np.where(np.squeeze(mask[i]) == 1)[0]])

    plot_correspond(imgs, title, kp, matches=inliers)

def SIFT_features(images, sift, bf):
    """
    :param images: list, list of all images
    :param sift: SIFT, SIFT detector instance
    :param bf: BFMatcher, BFMatcher instance
    :return:
        keypoints: list, list of all the keypoints of the image
        descriptors: list, list of all the descriptors of the image
        good_matches: list, list of all the good matches found between 2 sequential images
    """
    keypoints = []
    descriptors = []

    good_matches = []

    for i in range(len(images)):
        # Find the keypoints and descriptors with SIFT
        kp, des = sift.detectAndCompute(images[i], None)
        keypoints.append(kp)
        descriptors.append(des)

        # Find 2-nearest neighbours
        if (i > 0):
            matches = bf.knnMatch(descriptors[i - 1], des, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append([m])

            good_matches.append(good)

    return keypoints, descriptors, good_matches

def find_homography(selected_src, selected_dst):
    """
    Find the final homography using Least Squares Error
    :param selected_src: np.array(float), inliners from the source image
    :param selected_dst: np.array(float), inliners from the destination image
    :return:
        a: np.array(float), estimated homography
    """
    # Create M matrix
    M = []
    for indx in range(selected_dst.shape[0]):
        mx_nx = selected_src[indx][0] * selected_dst[indx][0]
        my_nx = selected_src[indx][1] * selected_dst[indx][0]
        x_row = np.asarray([selected_src[indx][0], selected_src[indx][1], 1, 0, 0, 0, -mx_nx, -my_nx], dtype=np.single)

        mx_ny = selected_src[indx][0] * selected_dst[indx][1]
        my_ny = selected_src[indx][1] * selected_dst[indx][1]
        y_row = np.asarray([0, 0, 0, selected_src[indx][0], selected_src[indx][1], 1, -mx_ny, -my_ny], dtype=np.single)

        # Add rows to M
        M.append(x_row)
        M.append(y_row)

    M = np.array(M, dtype=np.single)
    # Calculate inverse
    M_inv = np.linalg.pinv(M)
    # Find final homography
    a = np.hstack([np.dot(M_inv, selected_dst.flatten()), 1]).reshape(3, 3)

    return a

def ransac(src_points, dst_points, ransac_reproj_threshold=2, max_iters=500, inlier_ratio=0.8):
    """
    Calculate the set of inlier correspondences w.r.t. homography transformation, using the
    RANSAC method.
    :param src_points: numpy.array(float), coordinates of the points in the source image
    :param dst_points: numpy.array(float), coordinates of the points in the destination image
    :param ransac_reproj_threshold: float, maximum allowed reprojection error to treat a point pair
    as an inlier
    :param max_iters: int, the maximum number of RANSAC iterations
    :param inlier_ratio: float, ratio of inliers w.r.t. total number of correspondences
    :return:
    H: numpy.array(float), the estimated homography transformation using linear least-squares
    mask: numpy.array(uint8), mask that denotes the inlier correspondences
    """

    # Sanity checks
    assert src_points.shape == dst_points.shape
    assert ransac_reproj_threshold >= 0
    assert max_iters > 0
    assert inlier_ratio >= 0 and inlier_ratio <= 1

    final_inliners_indx = []

    cutoff = int(np.floor(inlier_ratio * (src_points.shape[0] - 4)))

    for i in range(max_iters):
        #Get 4 random points from the src and dst images
        random_indx = np.random.randint(src_points.shape[0], size=4)
        random_cor_src = src_points[random_indx, :]
        random_cor_dst = dst_points[random_indx, :]

        #Calculate estimated homography
        homography = cv2.getPerspectiveTransform(random_cor_src, random_cor_dst)

        inliners_indx = [entry for entry in random_indx]

        #Find inliners
        for p_idx in range(src_points.shape[0]):
            if (p_idx in random_indx):
                continue

            pix = src_points[p_idx]
            # Create homogeneous coordinates
            homography_coord = np.hstack([pix, 1])
            # Transform points with homography
            hom_x = np.dot(homography[0],homography_coord) / np.dot(homography[2],homography_coord)
            hom_y = np.dot(homography[1],homography_coord) / np.dot(homography[2],homography_coord)

            #Check if inliner
            x = (dst_points[p_idx,0] - hom_x) ** 2
            y =  (dst_points[p_idx,1] - hom_y) ** 2
            dist = np.sqrt(x +y)

            if (dist <= ransac_reproj_threshold):
                inliners_indx.append(p_idx)

        #Update the final variables if a better homography is found
        if (len(inliners_indx) >= len(final_inliners_indx)):
            final_inliners_indx = inliners_indx

            if (len(inliners_indx) >= cutoff):
                break

    #Calculate final homography using Least Squares
    inliners_src = src_points[final_inliners_indx, :]
    inliners_dst = dst_points[final_inliners_indx, :]

    H = find_homography(inliners_src, inliners_dst)
    mask = np.array([[1] if p_idx in final_inliners_indx else [0] for p_idx in range(src_points.shape[0])])

    return H, mask

def blend_images(src,dest, H):
    panorama_height = np.maximum(src.shape[0], dest.shape[0])
    panorama_width = src.shape[1] + dest.shape[1]
    panorama = np.zeros((panorama_height, panorama_width, 3), dtype=np.uint8)
    panorama[0:src.shape[0], 0:src.shape[1]] = src
    warped_img = cv2.warpPerspective(dest, H, (panorama_width, panorama_height), flags=cv2.WARP_INVERSE_MAP)

    # Blending
    temp_panorama = np.round(0.5 * panorama + 0.5 * warped_img).astype(np.uint8)
    temp_panorama[warped_img == [0, 0, 0]] = panorama[warped_img == [0, 0, 0]]
    temp_panorama[panorama == [0, 0, 0]] = warped_img[panorama == [0, 0, 0]]
    panorama = temp_panorama.copy()

    return panorama

def trim_panorama(panorama):
    #Remove the black space from the right part of the image
    keep_col = []
    for col in range(panorama.shape[1]):
        if (np.any(panorama[:,col] != [0,0,0])):
            keep_col.append(col)
    keep_col = np.array(keep_col)

    #The image does not have black space around it
    if (keep_col.shape[0] == panorama.shape[1]):
        return panorama

    trimmed_pan = panorama[:,keep_col.min() : keep_col.max()]

    #Remove black from top of the image
    has_black = True
    while (has_black):
        column_has_black = np.any(trimmed_pan[:, 0] == [0, 0, 0])
        row_has_black = np.any(trimmed_pan[0, :] == [0, 0, 0])
        if (column_has_black and row_has_black):
            trimmed_pan = trimmed_pan[1:, : trimmed_pan.shape[1]-1]
        if (column_has_black == False or row_has_black == False):
            has_black = False
            break

    #Remove black from the bottom of the image
    remove_row = -1
    for row in range(trimmed_pan.shape[0]):
        if (np.any(trimmed_pan[row,:] == [0,0,0])):
            remove_row = row
            break

    trimmed_pan = trimmed_pan[:remove_row, :]

    return trimmed_pan