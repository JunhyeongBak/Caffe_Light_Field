import caffe
import numpy as np
import cv2

class InputShiftingLayer(caffe.Layer):
    def setup(self, bottom, top):
        # Read params
        params = eval(self.param_str)
        self.tx = params['tx']
        self.ty = params['ty']
    
    def reshape(self, bottom, top):
        # Assign top blob shape
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
        # Get shift value matrix
        M = np.float32([[1, 0, self.tx], [0, 1, self.ty]])

        if bottom[0].data.shape[1] != 3:
            for b in range(bottom[0].data.shape[0]):
                top[0].data[b, 0, :, :] = cv2.warpAffine(bottom[0].data[b, 0, :, :], M, (bottom[0].data.shape[3], bottom[0].data.shape[2]),
                                                        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        else:
            for b in range(bottom[0].data.shape[0]):
                for c in range(3):
                    top[0].data[b, c, :, :] = cv2.warpAffine(bottom[0].data[b, c, :, :], M, (bottom[0].data.shape[3], bottom[0].data.shape[2]),
                                                            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)

    def backward(self, top, propagate_down, bottom):
        # Transfer gradient
        top[0].diff = bottom[0].diff