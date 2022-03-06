import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
from matplotlib import pyplot as plt

#Add imports if needed:
    """
    Your code here
    """
#end imports

#Add extra functions here:
    """
    Your code here
    """
#Extra functions end

# HW functions:
def getPoints(im1,im2,N):
    """
    Your code here
    """
    return p1,p2


def computeH(p1, p2):
    assert (p1.shape[1] == p2.shape[1])
    assert (p1.shape[0] == 2)
    """
    Your code here
    """
    return H2to1

def warpH(im1, H, out_size):
    """
    Your code here
    """
    return warp_im1

def imageStitching(img1, wrap_img2):
    """
    Your code here
    """
    return panoImg

def ransacH(matches, locs1, locs2, nIter, tol):
    """
    Your code here
    """
    return bestH

def getPoints_SIFT(im1,im2):
    """
    Your code here
    """
    return p1,p2

if __name__ == '__main__':
    print('my_homography')
    im1 = cv2.imread('data/incline_L.png')
    im2 = cv2.imread('data/incline_R.png')
    """
    Your code here
    """
