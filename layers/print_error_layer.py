import caffe
import numpy as np

class PrintErrorLayer(caffe.Layer):
    def setup(self, bottom, top):
        pass
    
    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].data.shape)         

    def forward(self, bottom, top):
        print('loss_x : ', bottom[0].data)
        print('loss_y : ', bottom[1].data)
        print('loss_tot : ', bottom[2].data)
        top[0].data[...] = bottom[0].data[...]
    
    def backward(self, top, propagate_down, bottom):
        pass