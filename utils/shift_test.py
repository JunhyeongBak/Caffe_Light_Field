import numpy as np
import cv2
import glob
import imageio
import warnings

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

def shift_value(i, shift_value):
    if i<=7:
        tx = 3*shift_value
    elif i>7 and i<=15:
        tx = 2*shift_value
    elif i>15 and i<=23:
        tx = 1*shift_value
    elif i>23 and i<=31:
        tx = 0
    elif i>31 and i<=39:
        tx = -1*shift_value
    elif i>39 and i<=47:
        tx = -2*shift_value
    elif i>47 and i<=55:
        tx = -3*shift_value
    else:
        tx = -4*shift_value
    
    if i==0 or (i%8==0 and i>7):
        ty = 3*shift_value
    elif i == 1 or (i-1)%8==0:
        ty = 2*shift_value
    elif i == 2 or (i-2)%8==0:
        ty = 1*shift_value
    elif i == 3 or (i-3)%8==0:
        ty = 0
    elif i == 4 or (i-4)%8==0:
        ty = -1*shift_value
    elif i == 5 or (i-5)%8==0:
        ty = -2*shift_value
    elif i == 6 or (i-6)%8==0:
        ty = -3*shift_value
    else:
        ty = -4*shift_value
    
    print(-tx, -ty)
    return -tx, -ty

if __name__ == '__main__':
    img27 = cv2.imread('/docker/etri/data/FlowerLF/27/16.png', cv2.IMREAD_GRAYSCALE)
    img27 = cv2.putText(img27, 'Center', (50, 50), cv2.FONT_ITALIC, 2, 255, 2)
    img3 = cv2.imread('/docker/etri/data/FlowerLF/30/16.png', cv2.IMREAD_GRAYSCALE)

    tx, ty = shift_value(27, 0.9)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    img27_shift = cv2.warpAffine(img27, M, (256, 192), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    

    img_gif = np.zeros((2, 192, 256))
    img_gif[0, :, :] = img27_shift
    img_gif[1, :, :] = img3

    cv2.imwrite('/docker/etri/data/FlowerLF/left_test2.png', img_gif[0, :, :])
    cv2.imwrite('/docker/etri/data/FlowerLF/right_test2.png', img_gif[1, :, :])
    #cv2.imwrite('/docker/etri/data/FlowerLF/left_test.png', img_gif[0, :, :])
    #cv2.imwrite('/docker/etri/data/FlowerLF/right_test.png', img_gif[1, :, :])
    imageio.mimsave('/docker/etri/data/FlowerLF/shift_test2.gif', img_gif, duration=0.5)

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
"""