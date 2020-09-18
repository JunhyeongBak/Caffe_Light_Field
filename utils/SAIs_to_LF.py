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

#LF Parameter
ANGULAR_RES_X = 5
ANGULAR_RES_Y = 5
IMG_HEIGHT = 960*3
IMG_WIDTH = 960*3
SPATIAL_HEIGHT = int(IMG_HEIGHT / ANGULAR_RES_Y)
SPATIAL_WIDTH = int(IMG_WIDTH / ANGULAR_RES_X)
CH = 3
SAIS = 25

def trans_SAIs_to_LF(imgSAIs):
    imgLF = np.zeros((IMG_HEIGHT, IMG_WIDTH, CH))
    full_LF_crop = np.zeros((SPATIAL_HEIGHT, SPATIAL_WIDTH, CH, 5, 5))
    
    for ax in range(ANGULAR_RES_X):
        for ay in range(ANGULAR_RES_Y):
            full_LF_crop[:, :, :, ax, ay] = imgSAIs[:, :, :, 5*ax+ay]
    
    for ax in range(ANGULAR_RES_X):
        for ay in range(ANGULAR_RES_Y):
            resized2 = full_LF_crop[:, :, :, ay, ax]
            #resized2 = cv2.resize(resized2, dsize=(256, 192), interpolation=cv2.INTER_LINEAR) 
            imgLF[ay::ANGULAR_RES_Y, ax::ANGULAR_RES_X, :] = resized2
    return imgLF

def trans_order(imgSAIs):
    imgSAIs2 = np.zeros((imgSAIs.shape))
    for i in range(25):
        if i == 0 or (i)%5==0:
            imgSAIs2[:, :, :, i//5] = imgSAIs[:, :, :, i]
        elif i == 1 or (i-1)%5==0:
            imgSAIs2[:, :, :, (i-1)//5+5] = imgSAIs[:, :, :, i]
        elif i == 2 or (i-2)%5==0:
            imgSAIs2[:, :, :, (i-2)//5+5*2] = imgSAIs[:, :, :, i]
        elif i == 3 or (i-3)%5==0:
            imgSAIs2[:, :, :, (i-3)//5+5*3] = imgSAIs[:, :, :, i]
        elif i == 4 or (i-4)%5==0:
            imgSAIs2[:, :, :, (i-4)//5+5*4] = imgSAIs[:, :, :, i]
    return imgSAIs2

if __name__ == '__main__':
    imgSAIs = np.zeros((SPATIAL_HEIGHT, SPATIAL_WIDTH, CH, SAIS))
    for i in range(25):
        imgSAI = cv2.imread('./utils/input_data/sai3_'+str(i)+'.png', cv2.IMREAD_COLOR)
        imgSAI = cv2.resize(imgSAI, dsize=(192*3, 192*3), interpolation=cv2.INTER_AREA)
        imgSAIs[:, :, :, i] = imgSAI

    imgSAIs2 = trans_order(imgSAIs)
    imgLF = trans_SAIs_to_LF(imgSAIs2)

    cv2.imwrite('./utils/output_data/result_lf.jpg', imgLF)