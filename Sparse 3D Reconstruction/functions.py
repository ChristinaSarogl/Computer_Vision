import cv2
import numpy as np
import helper as hlp

def find_matches(img1, img2):
    """
    Find corresponding points from image1 to image 2
    :param img1: numpy.array(uint8), left image
    :param img2: numpy.array(uint8), right image
    :return:
        pts1, pts2: numpy.array(float64), corresponding points from image 1 to image 2
    """
    # Make images greyscale
    img1_grey = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_grey = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Initiate SIFT detector
    sift = cv2.SIFT.create()

    # Find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1_grey, None)
    kp2, des2 = sift.detectAndCompute(img2_grey, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good, pts1, pts2 = [], [], []
    threshold = 0.75
    for m, n in matches:
        if m.distance < threshold * n.distance:
            good.append([m])
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)
    good = np.array(good)

    #Visualization info
    vis_info = {
        "good": good,
        "keypoints": [kp1, kp2]
    }

    pts1 = np.vstack(pts1)
    pts2 = np.vstack(pts2)

    return pts1, pts2, vis_info

def fundamental_matrix_linear_system(pts1, pts2):
    """
    Create linear equations for estimating the fundamental matrix in matrix form
    :param pts1: numpy.array(float), an array Nx2 that holds the source image points
    :param pts2: numpy.array(float), an array Nx2 that holds the destination image points
    :return:
    A: numpy.array(float), an array Nx8 that holds the left side coefficients of the linear equations
    b: numpy.array(float), an array Nx1 that holds the right side coefficients of the linear equations
    """

    # Sanity checks
    assert pts1.shape[0] >= 8 and pts2.shape[0] >= 8
    assert len(pts1.shape) == 2 and len(pts2.shape) == 2
    assert pts1.shape[1] == 2 and pts2.shape[1] == 2

    total_p = pts1.shape[0]

    # Create A matrix
    u1_u2 = pts1[:,0] * pts2[:,0]
    v1_u2 = pts1[:,1] * pts2[:,0]
    u2 = pts2[:,0]
    u1_v2 = pts1[:,0] * pts2[:,1]
    v1_v2 = pts1[:,1] * pts2[:,1]
    v2 = pts2[:,1]
    u1 = pts1[:,0]
    v1 = pts1[:,1]

    A = np.hstack([u1_u2.reshape(total_p,1),
                   v1_u2.reshape(total_p,1),
                   u2.reshape(total_p,1),
                   u1_v2.reshape(total_p,1),
                   v1_v2.reshape(total_p,1),
                   v2.reshape(total_p,1),
                   u1.reshape(total_p,1),
                   v1.reshape(total_p,1)])

    # Create b matrix
    b = np.full((total_p,1),-1)

    return A, b

def compute_correspond_epilines(points, which_image, F):
    """
    For points in an image of a stereo pair, computes the corresponding epilines in the other image
    :param points: numpy.array(float), an array Nx2 that holds the image points
    :param which_image: int, index of the image (1 or 2) that contains the points
    :param F: numpy.array(float), fundamental matrix between the stereo pair
    :return:
    epilines: numpy.array(float): an array Nx3 that holds the coefficients of the corresponding
    epipolar lines
    """

    # Sanity checks
    assert len(points.shape) == 2
    assert points.shape[1] == 2
    assert which_image == 1 or which_image == 2
    assert F.shape[0] == 3 and F.shape[1] == 3

    epilines = []

    # Create homogeneous coordinates
    p_hom = np.asarray([np.hstack([p,1]) for p in points])

    # Transpose if working with the right image
    if (which_image == 2):
        F = F.T

    l = np.dot(F, p_hom.T)

    #Normaliza coefficients
    a = l[0] / np.sqrt((l[0]**2) + (l[1]**2))
    b = l[1] / np.sqrt((l[0]**2) + (l[1]**2))
    c = l[2] / np.sqrt((l[0]**2) + (l[1]**2))

    return np.asarray([a,b,c]).T

def ransac(src_points, dst_points, ransac_reproj_threshold=2, max_iters=500, inlier_ratio=0.8, normalize=False):
    """
    Calculate the set of inlier correspondences w.r.t. fundamental matrix, using the RANSAC method.
    :param src_points: numpy.array(float), an Nx2 array that holds the coordinates of the points in the
    source image
    :param dst_points: numpy.array(float), an Nx2 array that holds the coordinates of the points in the
    destination image
    :param ransac_reproj_threshold: float, maximum allowed reprojection error to treat a point-epiline pair
    as an inlier
    :param max_iters: int, the maximum number of RANSAC iterations
    :param inlier_ratio: float, ratio of inliers w.r.t. total number of correspondences
    :return:
    F: numpy.array(float), the estimated fundamental matrix using linear least-squares
    mask: numpy.array(uint8), mask that denotes the inlier correspondences
    """

    # Sanity checks
    assert src_points.shape == dst_points.shape
    assert ransac_reproj_threshold >= 0
    assert max_iters > 0
    assert inlier_ratio >= 0 and inlier_ratio <= 1

    np.random.seed(123)

    final_inliners_indx = []
    cutoff = int(np.floor(inlier_ratio * (src_points.shape[0])))

    for i in range(max_iters):
        # Get 8 random points from the src and dst images
        random_indx = np.random.randint(src_points.shape[0], size=8)
        rand_src = src_points[random_indx, :]
        rand_dst = dst_points[random_indx, :]

        # Normalize points
        if (normalize):
            rand_src, rand_dst, M1, M2 = points_normalization(rand_src, rand_dst)

        # Construct Linear Eq. System
        A, b = fundamental_matrix_linear_system(rand_src, rand_dst)

        # Solve for F, within a try-except block
        try:
            A_inv = np.linalg.inv(A)
            F = np.hstack([np.dot(A_inv, b.flatten()), 1]).reshape(3, 3)
        except Exception as e:
            if (type(e) == np.linalg.LinAlgError):
                continue
            else :
                raise Exception("Sorry, something went wrong!")

        # De-normalize F
        if (normalize):
            F = np.dot(M2.T, np.dot(F, M1))

        # Calculate epilines in dest img
        epilines = compute_correspond_epilines(src_points, 1, F)

        # Check inliers
        inliners_indx = np.copy(random_indx).tolist()

        for idx in range(dst_points.shape[0]):
            epiline = epilines[idx]
            point = dst_points[idx]

            distance = np.absolute((epiline[0]*point[0]) + (epiline[1]*point[1]) + epiline[2])

            # Check that distance < reproj_threshold
            if ((distance < ransac_reproj_threshold) and (idx not in inliners_indx)):
                inliners_indx.append(idx)

        # Update the indexes if a better fundamental matrix is found
        if (len(inliners_indx) >= len(final_inliners_indx)):
            final_inliners_indx = inliners_indx

            if (len(inliners_indx) >= cutoff):
                break

    #Calculate final fundamental matrix using Least Squares
    inliners_src = src_points[final_inliners_indx, :]
    inliners_dst = dst_points[final_inliners_indx, :]

    # Normalize points
    if (normalize):
        inliners_src, inliners_dst, M1, M2 = points_normalization(inliners_src, inliners_dst)

    A, b = fundamental_matrix_linear_system(inliners_src, inliners_dst)

    # Solve for F
    try:
        A_inv = np.linalg.pinv(A)
        F = np.hstack([np.dot(A_inv, b.flatten()), 1]).reshape(3, 3)
    except Exception as e:
        print(e)
        raise Exception("Sorry, something went wrong!")

    # De-normalize F
    if (normalize):
        F = np.dot(M2.T, np.dot(F, M1))

    mask = np.array([[1] if p_idx in final_inliners_indx else [0] for p_idx in range(src_points.shape[0])])

    return F, mask

def points_normalization(pts1, pts2):
    """
    Normalize points so that each coordinate system is located at the centroid of the image points and
    the mean square distance of the transformed image points from the origin should be 2 pixels
    :param pts1: numpy.array(float), an Nx2 array that holds the source image points
    :param pts2: numpy.array(float), an Nx2 array that holds the destination image point
    :return:
    pts1_normalized: numpy.array(float), an Nx2 array with the transformed source image points
    pts2_normalized: numpy.array(float), an Nx2 array with the transformed destination image points
    M1: numpy.array(float), an 3x3 array - transformation for source image
    M2: numpy.array(float), an 3x3 array - transformation for destination image
    """

    # Sanity checks
    assert len(pts1.shape) == 2 and len(pts2.shape) == 2
    assert pts1.shape[0] == pts2.shape[0]
    assert pts1.shape[1] == 2 and pts2.shape[1] == 2

    # Calculate translation matrix
    centroid1 = pts1.mean(axis=0)
    centroid2 = pts2.mean(axis=0)

    transl_mat1 = np.array([[1, 0, -centroid1[0]],
                            [0, 1, -centroid1[1]],
                            [0, 0, 1]])

    transl_mat2 = np.array([[1, 0, -centroid2[0]],
                            [0, 1, -centroid2[1]],
                            [0, 0, 1]])

    # Calculate scaling matrix
    dist_sum1 = np.sum(np.sum((pts1 - centroid1) ** 2, axis=1))
    s1 = np.sqrt((2 * pts1.shape[0]) / dist_sum1)

    dist_sum2 = np.sum(np.sum((pts2 - centroid2) ** 2, axis=1))
    s2 = np.sqrt((2 * pts2.shape[0]) / dist_sum2)

    scaling_mat1 = np.array([[s1 , 0, 0],
                            [0, s1, 0],
                            [0, 0, 1]])

    scaling_mat2 = np.array([[s2, 0, 0],
                             [0, s2, 0],
                             [0, 0, 1]])

    # Calculate M and M'
    M1 = np.dot(scaling_mat1, transl_mat1)
    M2 = np.dot(scaling_mat2, transl_mat2)

    # Create homogeneous coordinates
    pts1_hom = np.asarray([np.hstack([p, 1]) for p in pts1])
    pts2_hom = np.asarray([np.hstack([p, 1]) for p in pts2])

    # Normalize points
    q1 = np.dot(M1, pts1_hom.T).T
    q2 = np.dot(M2, pts2_hom.T).T

    # Move back to 2D coordinates
    pts1_normalized = np.asarray([q1[:,0] / q1[:,2], q1[:,1] / q1[:,2]]).T
    pts2_normalized = np.asarray([q2[:,0] / q2[:,2], q2[:,1] / q2[:,2]]).T

    return pts1_normalized, pts2_normalized, M1, M2


def triangulation(P1, pts1, P2, pts2):
    """
    Triangulate pairs of 2D points in the images to a set of 3D points
    :param P1: numpy.array(float), an array 3x4 that holds the projection matrix of camera 1
    :param pts1: numpy.array(float), an array Nx2 that holds the 2D points on image 1
    :param P2: numpy.array(float), an array 3x4 that holds the projection matrix of camera 2
    :param pts2: numpy.array(float), an array Nx2 that holds the 2D points on image 2
    :return:
    pts3d: numpy.array(float), an array Nx3 that holds the reconstructed 3D points
    """

    #Sanity checks
    assert len(pts1.shape) == 2 and len(pts2.shape) == 2
    assert pts1.shape[0] == pts2.shape[0]
    assert pts1.shape[1] == 2 and pts2.shape[1] == 2
    assert P1.shape == (3,4) and P2.shape == (3,4)

    pts3d = []

    for idx in range(pts1.shape[0]):
        u = pts1[idx, 0]
        v = pts1[idx, 1]
        u_ = pts2[idx, 0]
        v_ = pts2[idx, 1]

        # Construct A
        A = np.array([[P1[0, 0] - P1[2, 0] * u, P1[0, 1] - P1[2, 1] * u, P1[0, 2] - P1[2, 2] * u],
                      [P1[1, 0] - P1[2, 0] * v, P1[1, 1] - P1[2, 1] * v, P1[1, 2] - P1[2, 2] * v],
                      [P2[0, 0] - P2[2, 0] * u_, P2[0, 1] - P2[2, 1] * u_, P2[0, 2] - P2[2, 2] * u_],
                      [P2[1, 0] - P2[2, 0] * v_, P2[1, 1] - P2[2, 1] * v_, P2[1, 2] - P2[2, 2] * v_]])

        # Construct b
        b = np.array([[P1[2, 3] * u - P1[0, 3]],
                      [P1[2, 3] * v - P1[1, 3]],
                      [P2[2, 3] * u_ - P2[0, 3]],
                      [P2[2, 3] * u_ - P2[1, 3]]])

        #Solve linear equation system
        x = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)), A.T), b.flatten())

        #If z is negative then the point is behind the cameras
        if (x[2] < 0):
            return np.zeros((1,))
        else:
            pts3d.append(x)

    return np.asarray(pts3d).reshape(pts1.shape[0],3)
