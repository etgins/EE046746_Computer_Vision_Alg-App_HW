import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
from matplotlib import pyplot as plt
import my_homography as mh
#Add imports if needed:

#end imports

#Add functions here:

def im2im(full_img, ref_book, new_book):

    # resizing the new book to the old one
    new_book = cv2.resize(new_book, ref_book.shape[:2][::-1])
    points_ref_book, points_full_img = mh.getPoints((ref_book * 255).astype('uint8'), (full_img * 255).astype('uint8'), 4)
    # computing the homography
    H = mh.computeH(points_ref_book, points_full_img)
    warpedNewBook = cv2.warpPerspective(new_book, np.linalg.inv(H), (full_img.shape[1], full_img.shape[0]))
    panoImg = mh.imageStitching(warpedNewBook, full_img)
    return panoImg

#Functions end

# HW functions:
def create_ref(im_path):
    im = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB)
    plt.imshow(im)
    dtype = "float32"
    # asking the user to point the corners
    plt.title('choose the corners of the book: top left and clockwise')
    corners = plt.ginput(4)
    corners = np.stack(corners).astype(dtype)
    corners = corners.astype(dtype)

    # setting the width and the length of the book
    width = int(corners[2, 0] - corners[0, 0])
    height = int(corners[2, 1] - corners[0, 1])

    boundary = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=dtype)

    H = cv2.getPerspectiveTransform(corners, boundary)
    ref_image = cv2.warpPerspective(im, H, (width, height))
    return ref_image



if __name__ == '__main__':
    print('my_ar')

    # section 1
    # favourite_wrap = create_ref('my_data/favourite.jpg')
    # plt.imshow(favourite_wrap)
    # plt.show()

    #section 2
    book_warp = create_ref('my_data/new_book.jpg')
    plt.imshow(book_warp)
    cv2.imwrite('my_data/new_book_ref.jpg', cv2.cvtColor(book_warp, cv2.COLOR_RGB2BGR))
    plt.close()

    ref_book = cv2.imread('my_data/favourite_warp.jpg').astype('float32') / 255
    ref_book = cv2.cvtColor(ref_book, cv2.COLOR_BGR2RGB)
    full_img = cv2.imread('my_data/favourite_new_scene.jpg').astype('float32') / 255
    full_img = cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB)
    new_book = cv2.imread('my_data/new_book_ref.jpg').astype('float32') / 255
    combine = (im2im(full_img, ref_book, new_book) * 255).astype('uint8')
    combine = cv2.cvtColor(combine, cv2.COLOR_RGB2BGR)
    plt.imshow(combine)
    plt.show()
