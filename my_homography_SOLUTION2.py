import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
from scipy import interpolate
from matplotlib import pyplot as plt

#Add imports if needed:

import random
#end imports

#Add extra functions here:


def basePanoFunc(im1, im2, p1, p2, homography="RANSAC"):
    # calculating H
    if homography == 'RANSAC':
        H2to1 = ransacH(p1, p2)
    else:
        H2to1 = computeH(p1, p2)
    dtype = 'uint8'
    # getting the outsize
    outsize, H, x_offset, y_offset = get_outsize(im2, H2to1)

    # setting the offsets
    y_min = abs(min(0, y_offset))
    y_max = max(0, y_offset)
    x_min = abs(min(0, x_offset))
    x_max = max(0, x_offset)

    # warping the image
    warp_im2 = warpH(im2, H, outsize)

    # setting the boundaries
    X = max(x_min + im1.shape[1], x_max + warp_im2.shape[1])
    Y = max(y_min + im1.shape[0], y_max + warp_im2.shape[0])

    im1_big = np.zeros([Y, X, 3], dtype=np.uint8)
    im1_big[y_min:y_min + im1.shape[0], x_min:x_min + im1.shape[1]] = (im1 * 255).astype(dtype)
    im2_big = np.zeros([Y, X, 3], dtype=np.uint8)
    im2_big[y_max:y_max + warp_im2.shape[0], x_max:x_max + warp_im2.shape[1]] = warp_im2
    panoImg = imageStitching(im1_big, im2_big)
    return panoImg

def makeItPano(name, totImgs, method="SIFT", homography="RANSAC"):
    # choose the order of the images
    firstIm = totImgs // 2 + 1
    set_seq = [*list(range(firstIm + 1, totImgs + 1)), *list(range(1, firstIm)[::-1])]

    dtype1 = 'float32'
    dtype2 = 'uint8'
    form = ".jpg"

    # stitch images one by one
    im1 = cv2.imread("data/" + name + str(firstIm) + form).astype(dtype1) / 255
    # change the image to RGB
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    # resizing the image to make process faster
    im1 = cv2.resize(im1, (im1.shape[1] // 4, im1.shape[0] // 4))
    for i in set_seq:
        im2 = cv2.imread("data/" + name + str(i) + form).astype(dtype1) / 255
        # change the image to RGB
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
        # resizing the image to make process faster
        im2 = cv2.resize(im2, (im2.shape[1] // 3, im2.shape[0] // 3))
        # getting the points according to the method
        if method == 'SIFT':
            p1, p2 = getPoints_SIFT((im1 * 255).astype(dtype2), (im2 * 255).astype(dtype2))
        else:
            p1, p2 = getPoints(im1, im2)
        im1 = basePanoFunc(im1, im2, p1, p2, homography).astype(dtype1) / 255
    plt.imsave("my_data/" + name + "_set_" + method + "_" + homography + form, (im1 * 255).astype(dtype2),
               format='jpg')
    return im1

def get_outsize(im, H):
    Y, X, _ = im.shape
    boundary = np.asarray([[0, 0, 1], [0, Y - 1, 1], [X - 1, 0, 1], [X - 1, Y - 1, 1]])
    boundary = boundary.T
    boundary = H @ boundary
    boundary_idx = np.floor(boundary[:2] / (boundary[2] + 1e-18)).astype(np.int32)
    y_width = np.max(boundary_idx[1]) - np.min(boundary_idx[1])
    x_width = np.max(boundary_idx[0]) - np.min(boundary_idx[0])
    y_offset = np.min(boundary_idx[1])
    x_offset = np.min(boundary_idx[0])
    H = np.asarray([[1, 0, -x_offset], [0, 1, -y_offset], [0, 0, 1]]) @ H
    outsize = (y_width, x_width)

    return outsize, np.linalg.inv(H), x_offset, y_offset

#Extra functions end


# HW functions:


def getPoints(im1,im2,N=6):

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.suptitle('Left Image')
    plt.imshow(im1)
    plt.subplot(1, 2, 2)
    plt.suptitle('Right Image')
    plt.imshow(im2)
    plt.suptitle('Choose {} pairs, first point from the left image and second from the right one'.format(N))
    points = plt.ginput(2 * N)
    plt.close()
    p1 = np.array(points[::2]).T
    p2 = np.array(points[1::2]).T
    return p1, p2



def computeH(p1, p2):
    assert (p1.shape[1] == p2.shape[1])
    assert (p1.shape[0] == 2)

    N = p2.shape[1]

    hg = np.ones((1, N))
    p2 = np.concatenate((p2, hg))

    # setting the matrix
    A = np.zeros((N * 2, 9))
    A[::2, :3] = np.transpose(p2)
    A[::2, 6:9] = -p1[0, :].reshape(N, 1) * np.transpose(p2)
    A[1::2, 3:6] = np.transpose(p2)
    A[1::2, 6:9] = -p1[1, :].reshape(N, 1) * np.transpose(p2)

    # find the smallest eigen vector
    eigValue, eigVector = np.linalg.eig(A.T @ A)
    idx = np.abs(eigValue).argmin()
    s = eigVector[:, idx]
    H2to1 = s.reshape(3, 3)
    return H2to1


def warpH(im1, H, out_size):
    im1_warped = np.zeros((*(out_size), 3))

    X, Y = np.meshgrid(np.arange(out_size[1]), np.arange(out_size[0]))
    X = X.reshape(-1)
    Y = Y.reshape(-1)

    # make the point homogeneous
    p2 = np.concatenate((X.reshape(1, -1), Y.reshape(1, -1), np.ones((1, X.size), dtype='uint16')), axis=0)

    p1 = H @ p2
    epsilon = 1e-18

    # normalize the point
    p1 /= (p1[2, :] + epsilon)

    y = p1[1]
    x = p1[0]

    x_max = im1.shape[1]
    y_max = im1.shape[0]

    # getting the boundary
    boundary = np.where(~((x < 0) | (y < 0) | (x >= x_max) | (y >= y_max)))[0]

    # reshape due to new boundary
    x_in = x[boundary]
    x_out = X[boundary]
    y_in = y[boundary]
    y_out = Y[boundary]

    dtype = 'uint32'
    # points that are in integer indexes
    int_idx = (x_in % 1 == 0) & (y_in % 1 == 0)
    im1_warped[y_out[int_idx], x_out[int_idx], :] = im1[x_in[int_idx].astype(dtype),
                                                                  y_in[int_idx].astype(dtype), :]
    # we work on the points that is not integer
    interp_idx = np.where(~int_idx)[0]
    for rgb in range(im1.shape[2]):
        interp = interpolate.interp2d(np.arange(im1.shape[1]), np.arange(im1.shape[0]), im1[:, :, rgb], kind='linear', fill_value=0)
        im1_warped[y_out[interp_idx], x_out[interp_idx], rgb] = \
            np.array([float(interp(XX, YY)) for XX, YY in zip(x_in[interp_idx], y_in[interp_idx])])

    warp_im1 = im1_warped
    # change the range of the values to 0-255
    warp_im1 = (warp_im1 * 255).astype('uint8')

    return warp_im1

def imageStitching(img1, wrap_img2):
    # make the mask
    mask1 = np.expand_dims((img1[:, :, 0] > 0) | (img1[:, :, 1] > 0) | (img1[:, :, 2] > 0), axis=-1)
    panoImg = np.zeros(img1.shape, dtype='uint8')
    panoImg = np.add(img1 * mask1, wrap_img2 * (1 - mask1))

    return panoImg

def ransacH(p1, p2, nIter=15, tol=2):
    firstIter = True

    for i in range(nIter):
        index = random.sample(range(len(p1[0])), 4)
        p2_chosen = p2.T[index]
        p1_chosen = p1.T[index]
        H = computeH(p1_chosen.T, p2_chosen.T)

        point_length = len(p2[0])
        # make the point homogeneous
        p2_hg = np.vstack([p2, np.ones((1, point_length))])

        p1_new_hg = H @ p2_hg
        # normalize the homogeneous point
        p1_est = np.stack([p1_new_hg[0] / p1_new_hg[2], p1_new_hg[1] / p1_new_hg[2]])

        err = np.linalg.norm(p1_est - p1, axis=0)
        inline_vec = (err < tol)
        inlines = np.sum(inline_vec)

        # in the first iteration we go inside
        if (firstIter):
            bestH = H
            max_inlines = inlines
            firstIter = False
            continue

        if (inlines > max_inlines):
            bestH = H
            max_inlines = inlines

    return bestH

def getPoints_SIFT(im1, im2, N=10):
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(im1, None)
    kp2, des2 = sift.detectAndCompute(im2, None)
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    # take N best
    good.sort(key=lambda m: m.distance)
    good = good[: min(N, len(good))]

    # init p1, p2
    p2 = np.zeros((2, 1))
    p1 = np.zeros((2, 1))

    for x in good:
        p1 = np.append(p1, np.array(kp1[x.queryIdx].pt).reshape(-1, 1), axis=1)
        p2 = np.append(p2, np.array(kp2[x.trainIdx].pt).reshape(-1, 1), axis=1)
    return p1[:, 1:], p2[:, 1:]


if __name__ == '__main__':
    print('my_homography')
    im1 = plt.imread('data/incline_L.png')
    im2 = plt.imread('data/incline_R.png')

    # section 1.1
    # p1, p2 = getPoints(im1, im2, 6)
    # np.save('p1.npy', p1)
    # np.save('p2.npy', p2)
    # p1 = np.load('p1.npy')
    # p2 = np.load('p2.npy')

    # section 1.2
    # H2to1 = computeH(p1, p2)

    # section 1.3
    # outsize, H, x_offset, y_offset = get_outsize(im2, H2to1)
    # im2_warp = warpH(im2, H, outsize)
    # plt.figure()
    # plt.imshow(im2_warp)
    # plt.show()
    # cv2.imwrite('my_data/im2_warp.jpg', im2_warp)

    # section 1.4
    # y_min = abs(min(0, y_offset))
    # y_max = max(0, y_offset)
    # x_min = abs(min(0, x_offset))
    # x_max = max(0, x_offset)
    # X = max(x1_offset + im1.shape[1], x2_offset + im2_warp.shape[1])
    # Y = max(y1_offset + im1.shape[0], y2_offset + im2_warp.shape[0])
    #
    # im1_big = np.zeros([Y, X, 3], dtype=np.uint8)
    # im1_big[y1_offset:y1_offset + im1.shape[0], x1_offset:x1_offset + im1.shape[1]] = (im1 * 255).astype('uint8')
    # im2_big = np.zeros([Y, X, 3], dtype=np.uint8)
    # im2_big[y2_offset:y2_offset + im2_warp.shape[0], x2_offset:x2_offset + im2_warp.shape[1]] = im2_warp
    #
    # panoImg = imageStitching(im1_big, im2_big)
    # plt.figure()
    # plt.imshow(panoImg)
    # plt.show()


    # section 1.5
    # p1_sift, p2_sift = getPoints_SIFT((im1*255).astype('uint8'), (im2*255).astype('uint8'))
    # H2to1_sift = computeH(p1_sift, p2_sift)
    # outsize_sift, H_sift, x_offset_sift, y_offset_sift = get_outsize(im2, H2to1_sift)
    # im2_warp_sift = warpH(im2, H_sift, outsize_sift)
    # plt.figure()
    # plt.imshow(im2_warp_sift)
    # plt.show()
    #
    # x1_offset_sift = abs(min(0, x_offset_sift))
    # y1_offset_sift = abs(min(0, y_offset_sift))
    # x2_offset_sift = max(0, x_offset_sift)
    # y2_offset_sift = max(0, y_offset_sift)
    # X = max(x1_offset_sift + im1.shape[1], x2_offset_sift + im2_warp_sift.shape[1])
    # Y = max(y1_offset_sift + im1.shape[0], y2_offset_sift + im2_warp_sift.shape[0])
    #
    # im1_big_sift = np.zeros([Y, X, 3], dtype=np.uint8)
    # im1_big_sift[y1_offset_sift:y1_offset_sift + im1.shape[0], x1_offset_sift:x1_offset_sift + im1.shape[1]] = (im1 * 255).astype('uint8')
    # im2_big_sift = np.zeros([Y, X, 3], dtype=np.uint8)
    # im2_big_sift[y2_offset_sift:y2_offset_sift + im2_warp_sift.shape[0], x2_offset_sift:x2_offset_sift + im2_warp_sift.shape[1]] = im2_warp_sift
    # panoImgSIFT = imageStitching(im1_big_sift, im2_big_sift)
    # plt.figure()
    # plt.imshow(panoImgSIFT)
    # plt.show()

    # section 1.6
    # beach_manual_regular = makeItPano("beach", 5, method="manual", homography='regular')
    # beach_SIFT_regular = makeItPano("beach", 5, method="SIFT", homography='regular')
    # sintra_manual_regular = makeItPano("sintra", 5, method="manual", homography='regular')
    # sintra_SIFT_regular = makeItPano("sintra", 5, method="SIFT", homography='regular')

    # section 1.7
    # beach_manual_RANSAC = makeItPano("beach", 5, method="manual", homography='RANSAC')
    # beach_SIFT_RANSAC = makeItPano("beach", 5, method="SIFT", homography='RANSAC')
    # sintra_manual_RANSAC = makeItPano("sintra", 5, method="manual", homography='RANSAC')
    # sintra_SIFT_RANSAC = makeItPano("sintra", 5, method="SIFT", homography='RANSAC')


    # section 1.8
    # view_manual_regular = makeItPano("view", 3, method="SIFT", homography='regular')
    # plt.figure()
    # plt.imshow(sintra_SIFT_regular)
    # plt.show()
