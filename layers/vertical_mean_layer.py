import caffe
import numpy as np

class VerticalMeanLayer(caffe.Layer):
    def setup(self, bottom, top):
        pass
    
    def reshape(self, bottom, top):
        self.src = bottom[0].data[...]
        self.src_shape = bottom[0].data[...].shape # lf center sai's shape
        top[0].reshape(self.src_shape[0], self.src_shape[1], 1, self.src_shape[3]) # caffe shape

    def forward(self, bottom, top):
        top[0].data[...] = np.zeros((self.src_shape[0], self.src_shape[1], 1, self.src_shape[3]))
        top[0].data[...][0, 0, 0, :] = np.mean(self.src, axis=2)

    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[...] = top[0].diff[...]