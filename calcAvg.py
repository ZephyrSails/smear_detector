import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import sys
import os


def imgRead(fileName):
    img = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)
    # IMREAD_GRAYSCALE
    # IMREAD_COLOR
    # IMREAD_UNCHANGED
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float64)


def imgReadDelta(fileName):
    img = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)
    laplacian = cv2.Laplacian(img, cv2.CV_64F).astype(np.float64)
    # IMREAD_GRAYSCALE
    # IMREAD_COLOR
    # IMREAD_UNCHANGED
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return laplacian.astype(np.float64)


def calcDeltaAvg(file_names):
    img = imgReadDelta(file_names[0])
    img_sum = np.zeros_like(img, dtype=np.float64)
    for name in file_names:
        img = imgReadDelta(name)
        img_sum += img
    n = len(file_names)

    average = abs(img_sum) / float(n)
    return average


def calcAvg(file_names):
    img = imgRead(file_names[0])
    img_sum = np.zeros_like(img, dtype=np.float64)
    for name in file_names:
        img = imgRead(name)
        img_sum += img
    n = len(file_names)

    average = img_sum / float(n)
    return average


def main(argv):
    if len(argv) != 2:
        sys.exit(-1)

    path = argv[1]

    # avg = calcDeltaAvg(glob.glob(os.path.join(path, "*.jpg"))) * 20
    avg = calcAvg(glob.glob(os.path.join(path, "*.jpg")))
    plt.imshow(avg, cmap = 'gray')
    plt.colorbar()
    plt.show()


    if path[-1] == '/':
        path = path[:-1]
    basename = os.path.basename(path)
    np.save("%s_avg.npy" % basename, avg)


if __name__ == '__main__':
    main(sys.argv)
