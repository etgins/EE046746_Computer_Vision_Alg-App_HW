import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
from scipy import interpolate
from scipy.linalg import eigh
from matplotlib import pyplot as plt

#Add imports if needed:

#end imports

# Add extra functions here:


def get_outsize(im, H):
    y, x,_ = im.shape
    corners = np.array([[0, 0, 1], [0, y - 1, 1], [x - 1, 0, 1], [x - 1, y - 1, 1]]).T
    corners = H @ corners
    im_corners_idx = np.floor(corners[:2] / (corners[2] + 1e-16)).astype(np.int32)
    x_width = np.max(im_corners_idx[0]) - np.min(im_corners_idx[0])
    x_offset = np.min(im_corners_idx[0])
    y_width = np.max(im_corners_idx[1]) - np.min(im_corners_idx[1])
    y_offset = np.min(im_corners_idx[1])
    outsize = (y_width, x_width)
    H = np.array([[1, 0, -x_offset], [0, 1, -y_offset], [0, 0, 1]])@H
    return outsize, np.linalg.inv(H), x_offset, y_offset

#Extra functions end

# HW functions:

def getPoints(im1,im2,N):


    plt.ion()
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im1)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(im2)
    plt.axis('off')
    plt.suptitle('choose {} pairs of points. choose one from left and then one from right and repeat.'.format(N))
    points = plt.ginput(n=2 * N, timeout=0)
    plt.close()
    p1 = np.asarray(points[::2]).transpose()
    p2 = np.asarray(points[1::2]).transpose()
    return p1, p2


def computeH(p1, p2):
    assert (p1.shape[1] == p2.shape[1])
    assert (p1.shape[0] == 2)
    N = p1.shape[1]

    if N >= 4:
        hg = np.ones((1, N))
        p2 = np.concatenate((p2, hg))

        A = np.zeros((N * 2, 9))
        A[::2, :3] = np.transpose(p2)
        A[::2, 6:9] = -p1[0, :].reshape(N, 1) * np.transpose(p2)
        A[1::2, 3:6] = np.transpose(p2)
        A[1::2, 6:9] = -p1[1, :].reshape(N, 1) * np.transpose(p2)

        _, _, V = np.linalg.svd(A)
        H2to1 = V[-1].reshape([3, 3])
        #eigValue, eigVector = np.linalg.eig(A.T @ A)
        #idx = np.abs(eigValue).argmin()
        #s = eigVector[:, idx]
        #H2to1 = s.reshape(3, 3)
        return H2to1
    else:
        return None


def warpH(im1, H, out_size, interpolation='linear'):

    eps = 1e-16
    im1_warped = np.zeros((*(out_size), 3))
    dtype = 'uint16'
    x_out, y_out = np.meshgrid(np.arange(out_size[1]), np.arange(out_size[0]))
    x_out = x_out.reshape(-1)
    y_out = y_out.reshape(-1)

    p2 = np.concatenate((x_out.reshape(1, -1), y_out.reshape(1, -1), np.ones((1, x_out.size), dtype=dtype)), axis=0)

    p1 = H @ p2
    p1 /= p1[2, :] + eps  # don't divide by 0

    x_in = p1[0]
    y_in = p1[1]
    inbounds = np.where(~((x_in < 0) | (y_in < 0) | (x_in >= im1.shape[1]) | (y_in >= im1.shape[0])))[0]
    # remove the indices that fall out of the im1 shape
    x_out = x_out[inbounds]
    y_out = y_out[inbounds]
    x_in = x_in[inbounds]
    y_in = y_in[inbounds]

    # first, care about integer indices which are not need interpolation
    non_interp_idx = (x_in % 1 == 0) & (y_in % 1 == 0)
    im1_warped[y_out[non_interp_idx], x_out[non_interp_idx], :] = im1[x_in[non_interp_idx].astype('uint32'),
                                                                  y_in[non_interp_idx].astype('uint32'), :]
    # now for the interpolation part:
    interp_idx = np.where(~non_interp_idx)[0]
    for dim in range(im1.shape[2]):
        interp = interpolate.interp2d(np.arange(im1.shape[1]), np.arange(im1.shape[0]), im1[:, :, dim], kind=interpolation,
                          fill_value=0)
        im1_warped[y_out[interp_idx], x_out[interp_idx], dim] = \
            np.array([float(interp(XX, YY)) for XX, YY in zip(x_in[interp_idx], y_in[interp_idx])])
    # warp_im1 = color.lab2rgb(im1_warped)  # range of values [0, 1]
    warp_im1 = im1_warped
    warp_im1 = (warp_im1 / 255).astype('float32')  # range of values [0, 255]
    return warp_im1



def imageStitching(img1, wrap_img2):
    panoImg = np.zeros(img1.shape, dtype='uint8')
    im1_mask = np.expand_dims((img1[:, :, 0] > 0) | (img1[:, :, 1] > 0) | (img1[:, :, 2] > 0), axis=-1)
    panoImg = np.add(img1 * im1_mask, wrap_img2 * (1 - im1_mask))

    return panoImg

    # panoImg = np.zeros(img1.shape, dtype='uint8')
    # im1_mask = (img1[:, :, 0] > 0) | (img1[:, :, 1] > 0) | (img1[:, :, 2] > 0)
    # im2_wrap_mask = wrap_img2.sum(
    #     2) > 10  # (wrap_img2[:, :, 0] > 0) | (wrap_img2[:, :, 1] > 0) | (wrap_img2[:, :, 2] > 0)
    # panoImg[im1_mask] = img1[im1_mask]
    # panoImg[im2_wrap_mask] = wrap_img2[im2_wrap_mask]
    # return panoImg


    #panoImg = np.zeros(img1.shape)
    #panoImg = panoImg.astype(np.uint8)
    #mask1 = np.expand_dims((img1[:, :, 0] > 0) | (img1[:, :, 1] > 0) | (img1[:, :, 2] > 0), axis=1)
    #panoImg = np.add(mask1*img1, (1-mask1)*wrap_img2)
    #return panoImg

def ransacH(matches, locs1, locs2, nIter, tol):
    """
    Your code here
    """
    #return bestH

def getPoints_SIFT(im1,im2):
    """
    Your code here
    """
    return p1, p2

if __name__ == '__main__':
    print('my_homography')
    im1 = cv2.imread('data/incline_L.png')
    im2 = cv2.imread('data/incline_R.png')
    H2to1 = computeH(p1, p2)
    outsize, H, x_offset, y_offset = get_outsize(im2, H2to1)

    p1_check, p2_check = points_checking(im1, H2to1, 10)

    colors = np.arange(0, 10)
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(im1)
    axes[0].axis('off')
    axes[0].scatter(p1_check[0], p1_check[1], marker='+', c=colors, cmap='jet')
    axes[0].set_title('Selected Points in First Image')
    axes[1].imshow(im2)
    axes[1].axis('off')
    axes[1].scatter(p2_check[0], p2_check[1], marker='+', c=colors, cmap='jet')
    axes[1].set_title('Projected Points in Second Image')
    plt.tight_layout()
    plt.savefig('my_data/points_checking.jpg')


    p1, p2 = getPoints(im1, im2, 4)
    H2to1 = computeH(p1, p2)
    outsize, H, x_offset, y_offset = get_outsize(im2, H2to1)
    #warp_im2 = cv2.imread("my_data/im2_warp.jpg")
    warp_im2 = warpH(im2, H2to1, outsize)
    x1_offset = abs(min(0, x_offset))
    y1_offset = abs(min(0, y_offset))
    x2_offset = max(0, x_offset)
    y2_offset = max(0, y_offset)
    X = max(x1_offset + im1.shape[1], x2_offset + warp_im2.shape[1])
    Y = max(y1_offset + im1.shape[0], y2_offset + warp_im2.shape[0])

    im1_big = np.zeros([Y, X, 3], dtype=np.uint8)
    im1_big[y1_offset:y1_offset + im1.shape[0], x1_offset:x1_offset + im1.shape[1]] = (im1 / 255).astype('float32')
    im2_big = np.zeros([Y, X, 3], dtype=np.uint8)
    im2_big[y2_offset:y2_offset + warp_im2.shape[0], x2_offset:x2_offset + warp_im2.shape[1]] = warp_im2

    panoImg = imageStitching(im1_big, im2_big)
    plt.figure()
    plt.imshow(panoImg)
    plt.show()
    plt.imsave("my_data/incline_manual.jpg", panoImg.astype('uint8'), format='jpg')
    #im2_warp = warpH(im2, H, outsize)
    #plt.figure()
    #plt.imshow(im2_warp)
    #plt.savefig('my_data/im2_warp.jpg')


    # x1_offset = abs(min(0, x_offset))
    # y1_offset = abs(min(0, y_offset))
    # x2_offset = max(0, x_offset)
    # y2_offset = max(0, y_offset)
    # X = max(x1_offset + im1.shape[1], x2_offset + im2_warp.shape[1])
    # Y = max(y1_offset + im1.shape[0], y2_offset + im2_warp.shape[0])
    #
    # im1_big = np.zeros([Y, X, 3], dtype=np.uint8)
    # im1_big[y1_offset:y1_offset + im1.shape[0], x1_offset:x1_offset + im1.shape[1]] = (im1 * 255).astype('uint8')
    # im2_big = np.zeros([Y, X, 3], dtype=np.uint8)
    # im2_big[y2_offset:y2_offset + im2_warp.shape[0], x2_offset:x2_offset + im2_warp.shape[1]] = (im2_warp* 255).astype('float32')

    #im2_warp = cv2.imread("my_data/im2_warp.jpg")
    # plt.figure()
    # plt.imshow(panoImg)




