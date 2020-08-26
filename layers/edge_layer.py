import caffe
import numpy as np
import cv2

class EdgeLayer(caffe.Layer):
    def setup(self, bottom, top):
        pass
    
    def reshape(self, bottom, top):
        # Assign top blob shape
        top[0].reshape(*bottom[0].data.shape)            

    def forward(self, bottom, top):
        for b in range(bottom[0].data.shape[0]):
            for c in range(bottom[0].data.shape[1]):
                top[0].data[b, c, :, :] = cv2.Laplacian(np.uint8(bottom[0].data[b, c, :, :]), cv2.CV_8U, ksize=3)

    def backward(self, top, propagate_down, bottom):
        # Transfer gradient
        top[0].diff[...] = bottom[0].diff[...]