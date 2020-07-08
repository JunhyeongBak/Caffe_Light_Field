import caffe
import numpy as np

class LfMeanLayer(caffe.Layer):
    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        self.lf_shape = bottom[0].data[...].shape
        top[0].reshape(self.lf_shape[0], 1, self.lf_shape[2], self.lf_shape[3])

    def forward(self, bottom, top):
        top[0].data[...][:, 0, :, :] = np.mean(bottom[0].data[...][:, :, :, :], axis=1)
        print(top[0].data[...].shape)

    def backward(self, top, propagate_down, bottom):
        pass