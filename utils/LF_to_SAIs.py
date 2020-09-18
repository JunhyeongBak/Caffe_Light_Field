### Flower light-field data's structure ###

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

# +---+----+----+----+----+
# | 9  | 17 | 25 | 33 | 41 |
# +---+----+----+----+----+
# | 10 | 18 | 26 | 34 | 42 |
# +---+----+----+----+----+
# | 11 | 19 | 27 | 35 | 43 |
# +---+----+----+----+----+
# | 12 | 20 | 28 | 36 | 44 |
# +---+----+----+----+----+
# | 13 | 21 | 29 | 37 | 45 |
# +---+----+----+----+----+

# +---+----+----+----+----+
# | 0 |  5 | 10 | 15 | 20 |
# +---+----+----+----+----+
# | 1 |  6 | 11 | 16 | 21 |
# +---+----+----+----+----+
# | 2 |  7 | 12 | 17 | 22 |
# +---+----+----+----+----+
# | 3 |  8 | 13 | 18 | 23 |
# +---+----+----+----+----+
# | 4 |  9 | 14 | 19 | 24 |
# +---+----+----+----+----+

import numpy as np
import cv2
import glob
import imageio
import warnings

ANGULAR_RES_X = 14
ANGULAR_RES_Y = 14
ANGULAR_RES_TARGET = 3*3
IMG_WIDTH = 3584
IMG_HEIGHT = 2688
SPATIAL_HEIGHT = int(IMG_HEIGHT / ANGULAR_RES_Y)
SPATIAL_WIDTH = int(IMG_WIDTH / ANGULAR_RES_X)
CH_INPUT = 3
CH_OUTPUT = 3
SHIFT_VALUE = 0.8

def trans_LF_to_SAIs(lf):
    full_LF_crop = np.zeros((SPATIAL_HEIGHT, SPATIAL_WIDTH, 3, ANGULAR_RES_Y, ANGULAR_RES_X))

    for ax in range(ANGULAR_RES_X):
        for ay in range(ANGULAR_RES_Y):
            resized = lf[ay::ANGULAR_RES_Y, ax::ANGULAR_RES_X, :]
            resized2 = cv2.resize(resized, dsize=(SPATIAL_WIDTH, SPATIAL_HEIGHT), interpolation=cv2.INTER_LINEAR)
            full_LF_crop[:, :, :, ay, ax] = resized2

    middle_LF = full_LF_crop[:, :, :, 3:11, 3:11]

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

    center_view = middle_LF[:, :, :, 1, 1] # Wrong?

    return center_view, LF_stack, LF_grid

if __name__ == '__main__':
    imgLF = cv2.imread('/docker/lf_depth/utils/input_lf.png', cv2.IMREAD_COLOR)

    imgCenter, imgStack, _ = trans_LF_to_SAIs(imgLF)

    for i in range(64):
        cv2.imwrite('/docker/lf_depth/utils/sai{}.png'.format(str(i)), imgStack[:, :, 3*i:3*i+3])