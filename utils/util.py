import numpy as np
import cv2
import glob
import imageio
import warnings

#LF Parameter
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

def bgrs_to_lumas(path='/docker/etri/data/FlowerLF/*.png'):
    glob_path = glob.glob(path)
    
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

def process_LF_all(lf):
    full_LF_crop = np.zeros((SPATIAL_HEIGHT, SPATIAL_WIDTH, 1, ANGULAR_RES_Y, ANGULAR_RES_X))
    lf = np.expand_dims(lf, axis=-1)
    
    for ax in range(ANGULAR_RES_X):
        for ay in range(ANGULAR_RES_Y):
            resized = lf[ay::ANGULAR_RES_Y, ax::ANGULAR_RES_X, :]
            resized2 = cv2.resize(resized, dsize=(SPATIAL_WIDTH, SPATIAL_HEIGHT), interpolation=cv2.INTER_LINEAR)
            resized2 = np.expand_dims(resized2, axis=-1)
            full_LF_crop[:, :, :, ay, ax] = resized2

    # Take 3x3 LF on the middle, since the side part suffer from vignetting
    middle_LF = full_LF_crop[:, :, :, 3:11, 3:11]  # Take 3x3 LF in 5D

    # To visualize the 3x3 LF
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

    center_view = middle_LF[:, :, :, 1, 1]

    return center_view, LF_stack, LF_grid

def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    center_view, GT, grid = process_LF_all(img)

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


"""
if __name__ == '__main__':
    img_dir = glob.glob('/docker/etri/data/FlowerLF/lf_luma/*.png')

    lf_cnt = 0

    for img_path in img_dir:
        lf_center, lf_stack, lf_grid = load_img(img_path)

        print('lf_stack.shape', lf_stack.shape)

        for lf_idx in range(64):
            cv2.imwrite('/docker/etri/data/FlowerLF/{}/{}.png'.format(str(lf_idx), str(lf_cnt)), lf_stack[:, :, lf_idx])
            print(lf_cnt, '/docker/etri/data/FlowerLF/{}/{}.png'.format(str(lf_idx), str(lf_cnt))) 

        lf_cnt = lf_cnt + 1
"""


"""                
if __name__ == '__main__':
    lf_gif = np.zeros((1, 64, 192*10, 256*10))
    
    lf_center = cv2.imread('/docker/etri/data/FlowerLF/{}/{}.png'.format(str(27), str(0)), cv2.IMREAD_GRAYSCALE)
    lf_center = cv2.resize(lf_center, dsize=(256*10, 192*10), interpolation=cv2.INTER_AREA)

    lk_param = dict(winSize = (100, 100),
                maxLevel = 2,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 3))

    feature_params = dict(maxCorners = 100,
                        qualityLevel = 0.3,
                        minDistance = 1,
                        blockSize = 7)

    p0 = cv2.goodFeaturesToTrack(lf_center, mask = None, **feature_params)
    mask = np.zeros_like(np.concatenate((np.expand_dims(lf_center, axis=-1), np.expand_dims(lf_center, axis=-1), np.expand_dims(lf_center, axis=-1)), axis=-1))
    color = np.random.randint(0,255,(100,3))
    for lf_idx in range(64):
        trans = cv2.imread('/docker/etri/data/FlowerLF/{}/{}.png'.format(str(lf_idx), str(0)), cv2.IMREAD_GRAYSCALE)
        trans = cv2.resize(trans, dsize=(256*10, 192*10), interpolation=cv2.INTER_AREA)
        frame = np.concatenate((np.expand_dims(trans, axis=-1), np.expand_dims(trans, axis=-1), np.expand_dims(trans, axis=-1)), axis=-1)
        print('frame.shape', frame.shape)
        p1, st, err = cv2.calcOpticalFlowPyrLK(lf_center, trans, p0, None, **lk_param)
        
        good_new = p1[st==1]
        good_old = p0[st==1]
        
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
        img = cv2.add(frame, mask)

        cv2.imwrite('/docker/etri/data/FlowerLF/test{}.png'.format(str(lf_idx)), img)
        
        
        print('p1.shape', p1.shape)


        print(st.shape)
        print(err.shape)
        # lf_gif[:, lf_idx, :, :]
    
    warnings.filterwarnings(action='once')
    fps = 0.05
    imageio.mimsave('/docker/etri/data/FlowerLF/test.gif', lf_gif[0, :, :, :], duration=fps)
"""

"""
if __name__ == '__main__':
    frame1 = cv2.imread('/docker/etri/data/FlowerLF/{}/{}.png'.format(str(27), str(0)), cv2.IMREAD_GRAYSCALE)
    frame1 = cv2.resize(frame1, dsize=(256*10, 192*10), interpolation=cv2.INTER_AREA)
    prev = frame1
    hsv = np.zeros((frame1.shape[0], frame1.shape[1], 3))
    hsv[...,1] = 255

    for lf_idx in range(64):
        fram2 = cv2.imread('/docker/etri/data/FlowerLF/{}/{}.png'.format(str(lf_idx), str(0)), cv2.IMREAD_GRAYSCALE)
        fram2 = cv2.resize(fram2, dsize=(256*10, 192*10), interpolation=cv2.INTER_AREA)
        next = fram2
        flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        print('afdsfasdfas', ang.shape)
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(np.uint8(hsv), cv2.COLOR_HSV2BGR)

        next_rgb = cv2.cvtColor(next, cv2.COLOR_GRAY2BGR)
        resut = cv2.addWeighted(next_rgb, 0.8, rgb, 0.2, 0)
        cv2.imwrite('/docker/etri/data/FlowerLF/test{}.png'.format(str(lf_idx)), resut)
"""


if __name__ == '__main__':
    img_dir = glob.glob('/docker/etri/data/FlowerLF/lf_rgb/*.png')

    lf_cnt = 0
    
    for img_path in img_dir:
        print(img_path)
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
        img_luma = img_yuv[:, :, 0]
        #cv2.imwrite('/docker/etri/data/FlowerLF/lf_luma/luma{}.png'.format(str(lf_cnt)), img_luma)

        lf_center, lf_stack, lf_grid = process_LF_all(img_luma)

        print('lf_stack.shape', lf_stack.shape)

        for lf_idx in range(64):
            cv2.imwrite('/docker/etri/data/FlowerLF/{}/{}.png'.format(str(lf_idx), str(lf_cnt)), lf_stack[:, :, lf_idx])
            print(lf_cnt, '/docker/etri/data/FlowerLF/{}/{}.png'.format(str(lf_idx), str(lf_cnt))) 

        lf_cnt = lf_cnt + 1

        if lf_cnt == 2000:
            exit
        