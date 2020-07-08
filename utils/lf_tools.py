import cv2
import numpy as np

#LF Parameter
ANGULAR_RES_X = 14;
ANGULAR_RES_Y = 14;
ANGULAR_RES_TARGET = 8*8
IMG_WIDTH = 3584
IMG_HEIGHT = 2688
SPATIAL_HEIGHT = int(IMG_HEIGHT / ANGULAR_RES_Y)
SPATIAL_WIDTH = int(IMG_WIDTH / ANGULAR_RES_X)
CH_INPUT = 3
CH_OUTPUT = 3
SHIFT_VALUE = 0.8

#Training parameter
BATCH_SIZE = 1
TRAIN_SIZE = 1.0
LR_G =  0.001 # 0.001 0.0005 0.00146 0.0002
EPOCH = 150000
DECAY_STEP = 5000
DECAY_RATE = 0.90
LAMBDA_L1 = 10.0
LAMBDA_MV = 10.0
LAMBDA_TV = 1e-1
EPS = 1e-6

# +---+----+----+----+----+----+----+----+
# | 0 | 8  | 16 | 24 | 32 | 40 | 48 | 56 |
# +---+----+----+----+----+----+----+----+
# | 1 | 9  | 17 | 25 | 33 | 41 | 49 | 57 |
# +---+----+----+----+----+----+----+----+
# | 2 | 10 | 18 | 26 | 34 | 42 | 50 | 58 |
# +---+----+----+----+----+----+----+----+
# | 3 | 11 | 19 | 27 | 35 | 43 | 51 | 59 |
# +---+----+----+----+----+----+----+----+
# | 4 | 12 | 20 | 28 | 36 | 44 | 52 | 60 |
# +---+----+----+----+----+----+----+----+
# | 5 | 13 | 21 | 29 | 37 | 45 | 53 | 61 |
# +---+----+----+----+----+----+----+----+
# | 6 | 14 | 22 | 30 | 38 | 46 | 54 | 62 |
# +---+----+----+----+----+----+----+----+
# | 7 | 15 | 23 | 31 | 39 | 47 | 55 | 63 |
# +---+----+----+----+----+----+----+----+


def process_LF(lf):
    full_LF_crop = np.zeros((SPATIAL_HEIGHT, SPATIAL_WIDTH, 3, ANGULAR_RES_Y, ANGULAR_RES_X))
    for ax in range(ANGULAR_RES_X):
        for ay in range(ANGULAR_RES_Y):
            resized = lf[ay::ANGULAR_RES_Y, ax::ANGULAR_RES_X, :]
            resized2 = cv2.resize(resized, dsize=(SPATIAL_WIDTH, SPATIAL_HEIGHT), interpolation=cv2.INTER_LINEAR)
            full_LF_crop[:, :, :, ay, ax] = resized2

    # Take 8x8 LF on the middle, since the side part suffer from vignetting
    middle_LF = full_LF_crop[:, :, :, 3:11, 3:11]  # Take 8x8 LF in 5D

    # To visualize the 8x8 LF
    for ax in range(8):
        for ay in range(8):
            if ay == 0:
                y_img = middle_LF[:, :, :, ay, ax]
            else:
                y_img = np.concatenate((y_img, middle_LF[:, :, :, ay, ax]), 0)

            if ax == 0 and ay == 0:
                LF_stack = middle_LF[:, :, :, 0, 0]
            else:
                LF_stack = np.concatenate((LF_stack, middle_LF[:, :, :, ay, ax]), 2)
        if ax == 0:
            LF_grid = y_img
        else:
            LF_grid = np.concatenate((LF_grid, y_img), 1)
        y_img = middle_LF[:, :, :, ay, ax]

    center_view = middle_LF[:, :, :, 3, 3]
    return center_view, LF_stack, LF_grid

def load_image(addr):
    # cv2 load images as BGR, convert it to RGB
    img = cv2.imread(addr)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    center_view, GT, grid = process_LF(img)

    center_view = np.uint8(center_view)
    grid = np.uint8(grid)
    GT = np.uint8(GT)

    return center_view, GT, grid

def load_image_3x3(addr):
    # cv2 load images as BGR, convert it to RGB
    img = cv2.imread(addr)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    center_view, GT, grid = process_LF(img)

    center_view = np.uint8(center_view)
    grid = np.uint8(grid)
    GT = np.uint8(GT)

    # 18, 19, 20,   26, 27, 28,   34, 35, 36
    center_view = GT[:, :, 27*3:27*3+3]
    small_GT =  GT[:, :, 18*3:18*3+3]
    print(GT.shape)
    small_GT =  np.concatenate((small_GT, GT[:, :, 19*3:19*3+3]), 2)
    small_GT =  np.concatenate((small_GT, GT[:, :, 20*3:20*3+3]), 2)
    small_GT =  np.concatenate((small_GT, GT[:, :, 26*3:26*3+3]), 2)
    small_GT =  np.concatenate((small_GT, GT[:, :, 27*3:27*3+3]), 2)
    small_GT =  np.concatenate((small_GT, GT[:, :, 28*3:28*3+3]), 2)
    small_GT =  np.concatenate((small_GT, GT[:, :, 34*3:34*3+3]), 2)
    small_GT =  np.concatenate((small_GT, GT[:, :, 35*3:35*3+3]), 2)
    small_GT =  np.concatenate((small_GT, GT[:, :, 36*3:36*3+3]), 2)

    return center_view, small_GT


if __name__ == '__main__':
# +---+----+----+
# | 0 | 3 | 6 |
# +---+----+----+
# | 1 | 4 | 7 |
# +---+----+----+
# | 2 | 5 | 8 |
# +---+----+----+

    center_view, GT = load_image_3x3('./data/flower1_lf.png')
    center_view2, GT2, _t = load_image('./data/flower1_lf.png')
    print('shape of center_view: ', center_view.shape)
    print('shape of GT: ', GT.shape)
    print('shape of GT2: ', GT2.shape)
    cv2.imwrite('./data/test_resut1.png', center_view)
    cv2.imwrite('./data/test_resut2.png', GT[:, :, 4*3:4*3+3])
    test_center = GT[:, :, 7*3:7*3+3]
    diff = cv2.subtract(center_view, test_center)
    cv2.imwrite('./data/test_resut3.png', diff)
