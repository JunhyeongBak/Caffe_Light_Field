import numpy as np
import cv2
import glob

#LF Parameter
ANGULAR_RES_X = 14
ANGULAR_RES_Y = 14
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
def shift_value(i):
    if i<=7:
        tx = 3*SHIFT_VALUE
    elif i>7 and i<=15:
        tx = 2*SHIFT_VALUE
    elif i>15 and i<=23:
        tx = 1*SHIFT_VALUE
    elif i>23 and i<=31:
        tx = 0
    elif i>31 and i<=39:
        tx = -1*SHIFT_VALUE
    elif i>39 and i<=47:
        tx = -2*SHIFT_VALUE
    elif i>47 and i<=55:
        tx = -3*SHIFT_VALUE
    else:
        tx = -4*SHIFT_VALUE
    
    if i==0 or (i%8==0 and i>7):
        ty = 3*SHIFT_VALUE
    elif i == 1 or (i-1)%8==0:
        ty = 2*SHIFT_VALUE
    elif i == 2 or (i-2)%8==0:
        ty = 1*SHIFT_VALUE
    elif i == 3 or (i-3)%8==0:
        ty = 0
    elif i == 4 or (i-4)%8==0:
        ty = -1*SHIFT_VALUE
    elif i == 5 or (i-5)%8==0:
        ty = -2*SHIFT_VALUE
    elif i == 6 or (i-6)%8==0:
        ty = -3*SHIFT_VALUE
    else:
        ty = -4*SHIFT_VALUE
        
    return tx, ty57 |
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

def bgrs_to_lumas():
    glob_path = glob.glob('/docker/etri/data/FlowerLF/*.png')
    for img in glob_path:
        print(img)
        img_bgr = cv2.imread(img, cv2.IMREAD_COLOR)
        
        img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
        print('img_yuv.shape: ', img_yuv.shape)
        img_luma = img_yuv[:, :, 0]
        cv2.imwrite(img, img_luma)

def process_LF(lf):
    full_LF_crop = np.zeros((SPATIAL_HEIGHT, SPATIAL_WIDTH, 1, ANGULAR_RES_Y, ANGULAR_RES_X))
    lf = np.expand_dims(lf, axis=-1)
    
    for ax in range(ANGULAR_RES_X):
        for ay in range(ANGULAR_RES_Y):
            resized = lf[ay::ANGULAR_RES_Y, ax::ANGULAR_RES_X, :]
            resized2 = cv2.resize(resized, dsize=(SPATIAL_WIDTH, SPATIAL_HEIGHT), interpolation=cv2.INTER_LINEAR)
            resized2 = np.expand_dims(resized2, axis=-1)
            full_LF_crop[:, :, :, ay, ax] = resized2

    # Take 3x3 LF on the middle, since the side part suffer from vignetting
    my_crop = full_LF_crop[:, :, :, 3:11, 3:11]  # Take 3x3 LF in 5D
    
    middle_LF = np.zeros((my_crop.shape[0], my_crop.shape[1], my_crop.shape[2], 3, 3))
    
    middle_LF[:, :, :, 0, 0] = my_crop[:, :, :, 0, 0]
    middle_LF[:, :, :, 1, 0] = my_crop[:, :, :, 3, 0]
    middle_LF[:, :, :, 2, 0] = my_crop[:, :, :, 6, 0]
    middle_LF[:, :, :, 0, 1] = my_crop[:, :, :, 0, 3]
    middle_LF[:, :, :, 1, 1] = my_crop[:, :, :, 3, 3]
    middle_LF[:, :, :, 2, 1] = my_crop[:, :, :, 6, 3]
    middle_LF[:, :, :, 0, 2] = my_crop[:, :, :, 0, 6]
    middle_LF[:, :, :, 1, 2] = my_crop[:, :, :, 3, 6]
    middle_LF[:, :, :, 2, 2] = my_crop[:, :, :, 6, 6]

    # To visualize the 3x3 LF
    for ax in range(3):
        for ay in range(3):
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

    center_view = middle_LF[:, :, :, 1, 1]

    return center_view, LF_stack, LF_grid

def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    center_view, GT, grid = process_LF(img)

    return center_view, GT, grid

def norm_wide(input):
    return (input - 128) / 128

def norm_pos(input):
    return ((((input - 128) / 128) + 1) / 2)

def wide_to_pos(input): # deprocess
    return (input + 1) / 2

def pos_to_wide(input): # preprocess
    return (input * 2) - 1

def shift_value(i):
    if i<=2:
        tx = 1*SHIFT_VALUE
    elif i>2 and i<=5:
        tx = 0
    else:
        tx = -1*SHIFT_VALUE

    if i==0 or (i%3==0 and i>2):
        ty = 1*SHIFT_VALUE
    elif i == 1 or (i-1)%3==0:
        ty = 0
    else:
        ty = -1*SHIFT_VALUE
        
    return tx, ty

if __name__ == '__main__':
    # bgrs_to_lumas()
    center_view, GT, grid =  load_img('./data/FlowerLF/IMG_2190_eslf.png')
    print('center_view.shape: ', norm_wide(center_view).shape)
    print('GT.shape: ', norm_wide(GT).shape)
    print('grid.shape: ', norm_wide(grid).shape)




"""
# jpg image -> png image
src = cv2.imread('/docker/etri/data/unnamed.jpg', cv2.IMREAD_COLOR)
print('image shape : ', src.shape)
dst = cv2.resize(src, dsize=(246, 192), interpolation=cv2.INTER_AREA)
print('image shape : ', dst.shape)
cv2.imwrite('/docker/etri/data/flower2.png', dst)
print('image shape : ', dst.shape)
dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
cv2.imwrite('/docker/etri/data/flower_gray2.png', dst)
"""