import caffe
import numpy as np

class LfAdd2Layer(caffe.Layer):
    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        self.lf_shape = bottom[0].data[...].shape
        top[0].reshape(*self.lf_shape)

    def forward(self, bottom, top):
        top[0].data[...] = np.add(bottom[0].data[...]*10, bottom[1].data[...]*10)

    def backward(self, top, propagate_down, bottom):
        pass