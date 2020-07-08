import caffe
import numpy as np

class LfAxisMeanLayer(caffe.Layer):
    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.sais_id = params['sais_id']

    def reshape(self, bottom, top):
        self.lf_shape = bottom[0].data[...].shape
        print('adfasdfasdfasdfasdfasdfasdfdfasdfawdfasdfasdfd', self.lf_shape)
        top[0].reshape(self.lf_shape[0], 1, self.lf_shape[2], self.lf_shape[3])

    def forward(self, bottom, top):
        top[0].data[...][:, 0, :, :] = np.mean(bottom[0].data[...][:, (self.sais_id*8):(self.sais_id*8)+8, :, :], axis=1)
        print(top[0].data[...].shape)

    def backward(self, top, propagate_down, bottom):
        pass