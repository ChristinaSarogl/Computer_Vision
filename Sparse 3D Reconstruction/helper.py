import cv2
import numpy as np
import matplotlib.pyplot as plt

### PROVIDED FUNCTIONS START ###
def drawlines(img1, img2, lines, pts1, pts2):
    """
        Draw epilines
    :param img1: numpy.array(), draw epilines on
    :param img2: numpy.array(), draw points on
    :param lines: numpy.array(), epilines
    :param pts1: numpy.array(), corresponding points on img1
    :param pts2: numpy.array(), corresponding points on img2
    :return:
        img1, img2: numpy.array()
    """

    H, W, _ = img1.shape
    img1 = np.copy(img1)
    img2 = np.copy(img2)

    for coeff, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0, y0 = 0, int(-coeff[2]/coeff[1])
        x1, y1 = W, int(-(coeff[2]+coeff[0]*W)/coeff[1])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color, 1, cv2.LINE_AA)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1, cv2.LINE_AA)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1, cv2.LINE_AA)

    return img1, img2

def drawcorrespondences(img1, img2, pts1, pts2):
    """
        Draw correspondences
    :param img1: numpy.array(), draw points on
    :param img2: numpy.array(), draw points on
    :param pts1: numpy.array(), corresponding points on img1
    :param pts2: numpy.array(), corresponding points on img2
    :return:
        img1, img2: numpy.array()
    """

    img1 = np.copy(img1)
    img2 = np.copy(img2)

    for pt1, pt2 in zip(pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1, cv2.LINE_AA)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1, cv2.LINE_AA)

    return img1, img2


def plot_3d_points(pts3d, fig_title, xlim=(-1, 1), ylim=(-1, 1), zlim=(2, 7), elev=-170, azim=20, vertical_axis='y',
                   show=True):
    """
        Plot 3D points using matplotlib
    :param pts3d: numpy.array(float), an array Nx3 that holds the input 3D points
    :param fig_title: str, figure title
    :param xlim: tuple, x-axis view limits
    :param ylim: tuple, y-axis view limits
    :param zlim: tuple, z-axis view limits
    :param elev: float, the elevation angle in degrees rotates the camera above the plane pierced by the vertical axis
    :param azim: float, the azimuthal angle in degrees rotates the camera about the vertical axis
    :param vertical_axis: str, the axis to align vertically
    :param show: bool, show or draw figure
    :return:
        None
    """

    fig = plt.figure(fig_title)
    ax = fig.add_subplot(projection='3d')
    ax.scatter(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2], marker='o')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.view_init(elev=elev, azim=azim, vertical_axis=vertical_axis)

    if show:
        plt.show()
    else:
        plt.draw()

### PROVIDED FUNCTIONS END ###


### MY FUNCTIONS START ###
def plot_images_1_2(imgs, title):
    plt.figure(title)

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(imgs[0], cv2.COLOR_BGR2RGB))
    plt.title("Image 1")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(imgs[1], cv2.COLOR_BGR2RGB))
    plt.title("Image 2")
    plt.axis("off")

def plot_correspond(imgs, title, kp, matches):
    plt.figure(title)
    plt.title(title)

    img_matches = np.empty((max(imgs[0].shape[0], imgs[1].shape[0]), imgs[0].shape[1] + imgs[1].shape[1], 3),
                           dtype=np.uint8)
    cv2.drawMatchesKnn(imgs[0], kp[0], imgs[1], kp[1], outImg=img_matches, matches1to2=matches, flags=2)
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.axis("off")

def plot_inliners(imgs, title, kp, matches):
    plt.figure(title)

    plt.subplot(2, 1, 1)
    img1 = np.empty((max(imgs[0].shape[0], imgs[1].shape[0]), imgs[0].shape[1] + imgs[1].shape[1], 3),
                           dtype=np.uint8)
    cv2.drawMatchesKnn(imgs[0], kp[0], imgs[1], kp[1], outImg=img1, matches1to2=matches[0], flags=2)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.title("Mine")
    plt.axis("off")

    plt.subplot(2, 1, 2)
    img2 = np.empty((max(imgs[0].shape[0], imgs[1].shape[0]), imgs[0].shape[1] + imgs[1].shape[1], 3),
                    dtype=np.uint8)
    cv2.drawMatchesKnn(imgs[0], kp[0], imgs[1], kp[1], outImg=img2, matches1to2=matches[1], flags=2)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.title("OpenCv")
    plt.axis("off")

def plot_epilines(imgs, title):
    plt.figure(title)

    plt.subplot(2,2,1)
    plt.imshow(cv2.cvtColor(imgs[0], cv2.COLOR_BGR2RGB))
    plt.title("Image 1")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(imgs[1], cv2.COLOR_BGR2RGB))
    plt.title("Image 2")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(imgs[2], cv2.COLOR_BGR2RGB))
    plt.title("Image 1 - OpenCV")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(imgs[3], cv2.COLOR_BGR2RGB))
    plt.title("Image 2 - OpenCV")
    plt.axis("off")

### MY FUNCTIONS END ###