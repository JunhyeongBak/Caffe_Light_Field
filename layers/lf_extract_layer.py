import caffe
import numpy as np
import cv2

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

#LF Parameter
ANGULAR_RES_X = 14
ANGULAR_RES_Y = 14
ANGULAR_RES_TARGET = 3*3
IMG_HEIGHT = 2688
IMG_WIDTH = 3584
SPATIAL_HEIGHT = int(IMG_HEIGHT / ANGULAR_RES_Y) # 192
SPATIAL_WIDTH = int(IMG_WIDTH / ANGULAR_RES_X) # 256
CH_INPUT = 1
CH_OUTPUT = 1
SHIFT_VALUE = 0.8

def my_process_LF(lf): 
    full_LF_crop = np.zeros((SPATIAL_HEIGHT, SPATIAL_WIDTH, 3, ANGULAR_RES_Y, ANGULAR_RES_X))
    
    for ax in range(ANGULAR_RES_X):
        for ay in range(ANGULAR_RES_Y):
            resized = lf[ay::ANGULAR_RES_Y, ax::ANGULAR_RES_X, :]
            resized2 = cv2.resize(resized, dsize=(SPATIAL_WIDTH, SPATIAL_HEIGHT), interpolation=cv2.INTER_LINEAR)
            full_LF_crop[:, :, :, ay, ax] = resized2
            
    my_crop = full_LF_crop[:, :, :, 3:11, 3:11]
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

class LfExtractLayer(caffe.Layer):
    def setup(self, bottom, top):
        if len(bottom) != 1:
            raise Exception("Input size error.")
        
        params = eval(self.param_str)
        self.debug = params['debug']

    def reshape(self, bottom, top):
        self.src = bottom[0].data

        top[0].reshape(self.src.shape[0], SPATIAL_HEIGHT, SPATIAL_WIDTH, self.src.shape[3]) # data(center veiw)
        top[1].reshape(self.src.shape[0], SPATIAL_HEIGHT, SPATIAL_WIDTH, self.src.shape[3]*ANGULAR_RES_TARGET) # label(SAI)

    def forward(self, bottom, top):
        top[0].data[...], top[1].data[...], _ = my_process_LF(self.src)

        if self.debug == 1:
            print('top[0].data[...].shape : ', top[0].data[...].shape)
            print('top[1].data[...].shape : ', top[1].data[...].shape)

    def backward(self, bottom, top):
        pass