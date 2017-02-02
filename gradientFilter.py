import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import sys
import os


def gradientMap(average):
    """
    This method is not used
    """
    sobelx = cv2.Sobel(average, cv2.CV_64F, 1, 0, ksize = 5)

    mask = np.zeros_like(sobelx, dtype=np.uint8)
    mask[sobelx < 60] = 255

    sobely = cv2.Sobel(average, cv2.CV_64F, 0, 1, ksize = 5)

    plt.imshow(sobely, cmap = 'gray')

    gX = abs(sobelx.sum())
    gY = abs(sobely.sum())
    # gX, gY = sobelx[x, y], sobely[x, y]
    divisor = (gX ** 2 + gY ** 2) ** 0.5
    gX /= divisor * 2.0
    gY /= divisor * 2.0
    # print gX, gY
    #
    deGradient = np.array(average)

    plt.imshow(average, cmap = 'gray')
    plt.show()


def laplacianGradient(avg):
    mask = np.zeros_like(avg, dtype = np.uint8)
    laplacian = cv2.Laplacian(avg, cv2.CV_64F, ksize = 31)

    # Gaussian blur
    blur = cv2.GaussianBlur(laplacian, (99, 99), 0)

    # Thresholding
    mask = np.zeros_like(blur, dtype = np.uint8)
    mask[blur > blur.max() * 0.4] = 255

    # fill holes
    kernel = np.ones((40, 40), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # # remove noise
    kernel = np.ones((30, 30), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # link the horizontal line
    kernel2 = np.ones((1, 100), np.uint8)
    mask = cv2.dilate(mask, kernel2, iterations = 1)
    mask = cv2.erode(mask, kernel2, iterations = 1)

    # remove large object (horizontal line)
    kernel3 = np.ones((1, 100), np.uint8)
    largeMask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3)
    mask -= largeMask
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


    plt.imshow(mask)
    plt.colorbar()
    plt.show()


def main(argv):
    # names = glob.glob("cam_3_avg.npy")
    names = glob.glob("*_avg.npy")
    for i, img in enumerate(names):
        average = np.load(img)
        plt.imshow(average, cmap = 'gray')
        laplacianGradient(average)


if __name__ == '__main__':
    main(sys.argv)
