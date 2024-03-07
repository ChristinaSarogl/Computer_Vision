import cv2
import numpy as np
import matplotlib.pyplot as plt

import helper as hlp
import functions as fun

def main():
    # Load left and right images
    print("Load images...\n")
    img1 = cv2.imread("data/image1.png", cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread("data/image2.png", cv2.IMREAD_UNCHANGED)

    hlp.plot_images_1_2([img1, img2], "Stereo Images")


    # Find SIFT matches
    print("Computer SIFT correspondences...\n")
    pts1, pts2, vis_info = fun.find_matches(img1, img2)

    hlp.plot_correspond([img1, img2], "SIFT Correspondences", vis_info["keypoints"], vis_info["good"])


    # Estimate fundamental matrix - My impl
    print("Estimate F (My implementation)...\n")
    F, mask = fun.ransac(pts1, pts2, 0.5, 5000, 0.9, normalize=True)
    pts1_in = pts1[mask.ravel() == 1]
    pts2_in = pts2[mask.ravel() == 1]


    # Estimate fundamental matrix - OpenCv
    print("Estimate F (OpenCV)...\n")
    pts1_cv = pts1.astype(np.int32)
    pts2_cv = pts2.astype(np.int32)
    F_cv, mask_cv = cv2.findFundamentalMat(pts1_cv, pts2_cv, cv2.FM_RANSAC, ransacReprojThreshold=0.5)
    pts1_in_cv = pts1_cv[mask_cv.ravel() == 1]
    pts2_in_cv = pts2_cv[mask_cv.ravel() == 1]


    # Plot inliner correspondences
    matches = [vis_info["good"][np.where(np.squeeze(mask) == 1)[0]],
               vis_info["good"][np.where(np.squeeze(mask_cv) == 1)[0]]]
    hlp.plot_inliners([img1, img2], "Inliner Correspondences", vis_info["keypoints"], matches=matches)


    # Calculate epipolar lines for both images
    print("Calculate epipolar lines...\n")
    # Points in image1 and draw lines on image 2
    lines2 = fun.compute_correspond_epilines(pts1_in, 1, F)
    lines2_cv = fun.compute_correspond_epilines(pts1_in_cv, 1, F_cv)

    # Points in image2 and draw lines on image 1
    lines1 = fun.compute_correspond_epilines(pts2_in, 2, F)
    lines1_cv = fun.compute_correspond_epilines(pts2_in_cv, 2, F_cv)


    # Plot epipolar lines
    image2_ep, image1_points = hlp.drawlines(img2, img1, lines2, pts2_in.astype(np.int32), pts1_in.astype(np.int32))
    image2_ep_cv, image1_points_cv = hlp.drawlines(img2, img1, lines2_cv, pts2_in_cv, pts1_in_cv)

    hlp.plot_epilines([image1_points, image2_ep, image1_points_cv, image2_ep_cv],
                      "Epipolar lines of Image 1 on Image 2")

    image1_ep, image2_points = hlp.drawlines(img1, img2, lines1, pts1_in.astype(np.int32), pts2_in.astype(np.int32))
    image1_ep_cv, image2_points_cv = hlp.drawlines(img1, img2, lines1_cv, pts1_in_cv, pts2_in_cv)

    hlp.plot_epilines([image1_ep, image2_points, image1_ep_cv, image2_points_cv],
                      "Epipolar lines of Image 2 on Image 1")


    #Load intrinsics matrices
    intrinsics = np.load("data/intrinsics.npz")
    K1 = intrinsics['K1']
    K2 = intrinsics['K2']


    # Estimate essential matrix for camera 1
    print("Estimate E...\n")
    E1 = np.dot(K2.T, np.dot(F, K1))
    E1_cv = np.dot(K2.T, np.dot(F_cv, K1))

    # Decompose essential matrices
    R1, R2, t = cv2.decomposeEssentialMat(E1)
    R1_cv, R2_cv, t_cv = cv2.decomposeEssentialMat(E1_cv)

    E2 = [np.hstack([R1,t]), np.hstack([R1,-t]), np.hstack([R2,t]), np.hstack([R2,-t])]

    E2_cv = [np.hstack([R1_cv,t_cv]), np.hstack([R1_cv,-t_cv]), np.hstack([R2_cv,t_cv]), np.hstack([R2_cv,-t_cv])]


    # Load good correspondences
    print("Load good correspondences... \n ")
    good = np.load("data/good_correspondences.npz")
    loaded_pts1 = good['pts1']
    loaded_pts2 = good['pts2']

    #Visualize correspondences
    loaded_img1, loaded_img2 = hlp.drawcorrespondences(img1, img2, loaded_pts1, loaded_pts2)
    hlp.plot_images_1_2([loaded_img1, loaded_img1], "Loaded correspondences")


    #Triagulation
    print("Performing triangulation...\n")

    P1 = np.dot(K1, np.hstack([np.eye(3), np.zeros((3, 1))]))

    for e2 in E2:
        P2 = np.dot(K2,e2)
        pts3d = fun.triangulation(P1, loaded_pts1, P2, loaded_pts2)

        # If the result's shape is equal to the points then all points are in front of both cameras
        if(pts3d.shape[0] == loaded_pts1.shape[0]):
            break

    for e2 in E2_cv:
        P2 = np.dot(K2,e2)
        pts3d_cv = fun.triangulation(P1, loaded_pts1, P2, loaded_pts2)

        # If the result's shape is equal to the points then all points are in front of both cameras
        if(pts3d_cv.shape[0] == loaded_pts1.shape[0]):
            break


    # Plot 3D points
    hlp.plot_3d_points(pts3d, "3D Reconstruction", show=False)
    hlp.plot_3d_points(pts3d_cv, "3D Reconstruction - OpenCV")

    print("DONE!\n")

if __name__ == "__main__":
    main()