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
import imageio

class VisualizationLayer(caffe.Layer):
    def setup(self, bottom, top):
        # Read params
        params = eval(self.param_str)
        self.mode = params['mode']
        self.weight = params['weight']
        self.path = params['path']
    
    def reshape(self, bottom, top):
        # Assign top blob shape
        top[0].reshape(*bottom[0].data.shape)            

    def forward(self, bottom, top):
        # Transfer original value
        top[0].data = bottom[0].data

        # Visualize input image
        if self.mode == 'img':
            if bottom[0].data.shape[1] != 1
                raise Exception("ERR - Image chennel should be 1ch")
            cv2.imwrite(self.path, top[0].data[0, 0, :, :]*self.weight)
        elif self.mode == 'gif':
            imageio.mimsave(self.path, top[0].data[0, :, :, :]*self.weight, duration=0.5)
        else:
            raise Exception("ERR - Select the correct mode")

    def backward(self, top, propagate_down, bottom):
        # Transfer gradient
        top[0].diff = bottom[0].diff

"""
if len(bottom) != 1:
            raise Exception("Input size error.")
"""