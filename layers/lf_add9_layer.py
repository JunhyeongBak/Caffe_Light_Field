import caffe
import numpy as np

class LfAdd9Layer(caffe.Layer):
    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        self.lf_shape = bottom[0].data[...].shape
        top[0].reshape(*self.lf_shape)

    def forward(self, bottom, top):
        top[0].data[...] = bottom[0].data[...]+bottom[1].data[...]+bottom[2].data[...]+bottom[3].data[...]+bottom[4].data[...]+bottom[5].data[...]+bottom[6].data[...]+bottom[7].data[...]+bottom[8].data[...]

    def backward(self, top, propagate_down, bottom):
        pass