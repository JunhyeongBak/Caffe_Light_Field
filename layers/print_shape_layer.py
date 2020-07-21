import caffe
import numpy as np

class PrintShapeLayer(caffe.Layer):
    def setup(self, bottom, top):
        # Read params
        params = eval(self.param_str)
        self.comment = params['comment']
    
    def reshape(self, bottom, top):
        # Assign top blob shape
        top[0].reshape(1)

    def forward(self, bottom, top):
        top[0].data[...] = 0
        print(self.comment+' : '+top[0].data.shape)

    def backward(self, top, propagate_down, bottom):
        # Transfer gradient
        top[0].diff = bottom[0].diff