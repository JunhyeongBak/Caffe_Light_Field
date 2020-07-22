import numpy as np
import cv2
import glob
import imageio
import warnings

#LF Parameter
ANGULAR_RES_X = 5
ANGULAR_RES_Y = 5
ANGULAR_RES_TARGET = 3*3
IMG_WIDTH = 3584
IMG_HEIGHT = 2688
SPATIAL_HEIGHT = int(IMG_HEIGHT / ANGULAR_RES_Y)
SPATIAL_WIDTH = int(IMG_WIDTH / ANGULAR_RES_X)
CH_INPUT = 3
CH_OUTPUT = 3
SHIFT_VALUE = 0.8

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

def process_LF_all(LF_stack):
    lf = np.zeros((960, 1280, 3))
    full_LF_crop = np.zeros((192, 256, 3, 5, 5))
    
    middle_LF = full_LF_crop[:, :, :, :, :]
    
    for ax in range(5):
        for ay in range(5):
            middle_LF[:, :, :, ax, ay] = LF_stack[:, :, :, 5*ax+ay]

    full_LF_crop[:, :, :, :, :] = middle_LF
    
    for ax in range(ANGULAR_RES_X):
        for ay in range(ANGULAR_RES_Y):
            resized2 = full_LF_crop[:, :, :, ay, ax]
            #resized = cv2.resize(resized2, dsize=(256, 192), interpolation=cv2.INTER_LINEAR) 
            lf[ay::ANGULAR_RES_Y, ax::ANGULAR_RES_X, :] = resized2
    
    return lf
            
if __name__ == '__main__':
    imgSAIs = np.zeros((192, 256, 3, 64))
    for i in range(64):
        imgSAI = cv2.imread('/docker/lf_depth/utils/sai'+str(i)+'.png', cv2.IMREAD_COLOR)
        imgSAIs[:, :, :, i] = imgSAI

    imgLF = process_LF_all(imgSAIs)

    cv2.imwrite('/docker/lf_depth/utils/lf.png', imgLF)
    #cv2.resize(resized2, dsize=(7574, 5264), interpolation=cv2.INTER_LINEAR)
    #cv2.imwrite('/docker/lf_depth/utils/blend.png', imgBlend)