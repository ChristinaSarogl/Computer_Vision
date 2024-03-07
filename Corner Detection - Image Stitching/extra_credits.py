import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS

def calc_function_vals(name):
    """
    Imports the image and calculates the focal length value in pixels
    :param name: String, name of the image file
    :return:
        f: float, focal length in pixels
        img: np.array(uint8), image imported using cv2
    """

    image = Image.open(name)
    img = cv2.imread(name, cv2.IMREAD_UNCHANGED)

    exifdata = image._getexif()
    properties = {}
    for tag_id in exifdata:
        # get the tag name, instead of human unreadable tag id
        tag = TAGS.get(tag_id, tag_id)
        data = exifdata.get(tag_id)
        properties[tag] = data

    focal_length = properties['FocalLengthIn35mmFilm']

    #Calcualte focal length in pixels
    f = (focal_length / 35) * img.shape[1]

    return f, img

def warp_cyl_coord():
    """
    Wrap the images within a cylindrical coordinate system
    :return:
        images: list, images within a cylindrical coordinate system
    """
    images = []
    for i in range(1, 6):
        name = "panoramas/pano_{}.jpg".format(i)

        f, img = calc_function_vals(name)

        xc = img.shape[1] / 2  # width
        yc = img.shape[0] / 2  # height

        angles = np.array([(x - xc) / f for x in range(img.shape[1])])
        height = np.array([(y - yc) / f for y in range(img.shape[0])])

        new_img = []
        for y in range(img.shape[0]):
            row = []
            for x in range(img.shape[1]):
                x_ = np.sin(angles[x])
                z_ = np.cos(angles[x])

                new_x = f * x_ / z_ + xc
                new_y = f * height[y] / z_ + yc

                #Insert black if the pixels coordinates are out of range
                if (new_x < 0 or new_y < 0):
                    row.append([0, 0, 0])
                elif (new_x > img.shape[1] or new_y > img.shape[0]):
                    row.append([0, 0, 0])
                else:
                    val = img[int(np.floor(new_y)), int(np.floor(new_x))]
                    row.append(val)

            new_img.append(row)

        new_img = np.array(new_img, dtype=np.uint8)
        images.append(new_img)

        print("EXTRA CREDITS:  Calculated cylindrical coordinates Image {}".format(i))

    return images