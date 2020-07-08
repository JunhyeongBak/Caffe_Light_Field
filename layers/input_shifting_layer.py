# LF matix
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

import caffe
import numpy as np
import cv2

SHIFT_VALUE = 0.8

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
    
    return -tx, -ty

class InputShiftingLayer(caffe.Layer):
    def setup(self, bottom, top):
        # Read params
        params = eval(self.param_str)
        self.id = params['id']
    
    def reshape(self, bottom, top):
        # Assign top blob shape
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
        # Get shift value matrix
        tx, ty = shift_value(self.id, SHIFT_VALUE)
        M = np.float32([[1, 0, tx], [0, 1, ty]])

        # Shift image
        top[0].data[0, 0, :, :] = cv2.warpAffine(bottom[0].data[0, 0, :, :], M, (bottom[0].data.shape[3], bottom[0].data.shape[2]),
                                                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    def backward(self, top, propagate_down, bottom):
        # Transfer gradient
        top[0].diff = bottom[0].diff