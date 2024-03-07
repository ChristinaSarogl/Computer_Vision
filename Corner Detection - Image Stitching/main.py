import cv2
import numpy as np
import matplotlib.pyplot as plt

import harris_corner as hc
import image_stiching as pan
import extra_credits as ec

def harris_corners():
    #Load images
    images=[]
    for i in range(1,7):
        name = "corners/corner_{}.png".format(i)
        images.append(cv2.imread(name, cv2.IMREAD_UNCHANGED))

    hc.plot_corners(images, "Images")

    quality_level, max_corners, min_distance, block_size, k = 0.01, 0, 10.0, 5, 0.05
    corners_list = []
    cv2_list = []

    for i in range(len(images)):
        # Convert to grayscale and float
        img_grey = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY).astype(np.float32)

        #Find corners
        img_corners = hc.detect_corners(img_grey, max_corners, quality_level, min_distance, block_size, k)
        corners_list.append(img_corners)

        cv2_corners = cv2.goodFeaturesToTrack(image=img_grey, maxCorners=max_corners, qualityLevel=quality_level,
                                              minDistance=min_distance, blockSize=block_size, useHarrisDetector=1, k=k)
        cv2_list.append(cv2_corners)

        print("HARRIS CORNER:  Found corners for image ", i + 1)

    hc.plot_corners(images, "Results", img_corners=corners_list)
    hc.plot_corners(images, "CV2", img_corners=cv2_list)

    plt.show()

def image_stiching(**kwargs):
    #Check if images are already passed to the function
    images = kwargs.get("images", None)

    if (images == None):
        # Load images
        images = []

        for i in range(1, 6):
            name = "panoramas/pano_{}.jpg".format(i)
            img = cv2.imread(name, cv2.IMREAD_UNCHANGED)
            images.append(img)

    # Make images greyscale
    greyscale = []
    for img in images:
        greyscale.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    pan.plot_images_1_5(images, "Panorama images")

    print("PANORAMA:  Loaded images")

    # Initiate SIFT detector
    sift = cv2.SIFT.create()

    # BFMatcher with default params
    bf = cv2.BFMatcher()

    #Find SIFT descriptors
    keypoints, descriptors, good_matches = pan.SIFT_features(greyscale, sift, bf)
    pan.plot_images_1_5(images, "SIFT features", keypoints=keypoints)

    print("\nPANORAMA:  Found SIFT correspondences for pair of images")

    #Plot correspondances
    pan.plot_correspond(images, "Good correspondences", keypoints, matches=good_matches)

    ransac_reprojection_threshold, max_iters = 1.0, 1000
    hom, hom_cv = [], []
    masks, masks_cv = [], []

    # Remove outliers
    for idx in range(len(images)-1):
        #Get the locations of the good SIFT features
        src = np.float32([keypoints[idx][g[0].queryIdx].pt for g in good_matches[idx]])
        dst = np.float32([keypoints[idx + 1][g[0].trainIdx].pt for g in good_matches[idx]])

        #Use RANSAC
        H, mask = pan.ransac(src, dst, ransac_reprojection_threshold, max_iters)

        #CV2 implementation
        H_cv, mask_cv = cv2.findHomography(srcPoints= src, dstPoints = dst, method = cv2.RANSAC,
            ransacReprojThreshold = ransac_reprojection_threshold, maxIters = max_iters)

        #Save homography between first two images
        if idx == 0:
            hom, hom_cv = H, H_cv

        masks.append(mask)
        masks_cv.append(mask_cv)

        print("\nPANORAMA:  Removed outliers from {} - {}".format(idx, idx + 1))


    pan.plot_no_outliers(images, "Remove outliers", good_matches, masks, keypoints)
    pan.plot_no_outliers(images, "Remove outliers OpenCV", good_matches, masks_cv, keypoints)

    #Stich images
    panorama = pan.blend_images(images[0], images[1], hom)
    for img in range(2,len(images)-1):
        panorama_grey = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors, good_matches = pan.SIFT_features([panorama_grey, greyscale[img]], sift, bf)

        src = np.float32([keypoints[0][g[0].queryIdx].pt for g in good_matches[0]])
        dst = np.float32([keypoints[1][g[0].trainIdx].pt for g in good_matches[0]])

        H, mask = pan.ransac(src, dst, ransac_reprojection_threshold, max_iters)

        panorama = pan.blend_images(panorama, images[img], H)

    print("\nPANORAMA:  Stitched panorama")

    #Stich images
    panorama_cv = pan.blend_images(images[0], images[1], hom_cv)
    for img in range(2, len(images) - 1):
        panorama_grey = cv2.cvtColor(panorama_cv, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors, good_matches = pan.SIFT_features([panorama_grey, greyscale[img]], sift, bf)

        src = np.float32([keypoints[0][g[0].queryIdx].pt for g in good_matches[0]])
        dst = np.float32([keypoints[1][g[0].trainIdx].pt for g in good_matches[0]])

        H_cv, mask_cv = cv2.findHomography(srcPoints=src, dstPoints=dst, method=cv2.RANSAC,
                                           ransacReprojThreshold=ransac_reprojection_threshold, maxIters=max_iters)

        panorama_cv = pan.blend_images(panorama_cv, images[img], H_cv)
    print("\nPANORAMA:  Stitched panorama OpenCV")

    panorama = pan.trim_panorama(panorama)
    panorama_cv = pan.trim_panorama(panorama_cv)

    plt.figure("Panorama")
    plt.subplot(2, 1, 1)
    plt.title("Mine")
    plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.subplot(2, 1, 2)
    plt.title("OpenCV")
    plt.imshow(cv2.cvtColor(panorama_cv, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.show()

def extra():
    images = ec.warp_cyl_coord()
    image_stiching(images= images)

def main():
    harris_corners()

    image_stiching()

    extra()

if __name__ == "__main__":
    main()