import caffe
import numpy as np
from datasetPreprocess import load_img, norm_wide, norm_pos, wide_to_pos, pos_to_wide

class MyImageDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        pass
        
    def reshape(self, bottom, top):
        top[0].reshape(1, 192, 256, 1)
        top[1].reshape(1, 192, 256, 9)
        
    def forward(self, bottom, top):
        center_view_a, inLF_a, grid_a =  load_img('/docker/etri/data/FlowerLF/IMG_2220_eslf.png')
        center_view_b, inLF_b, grid_b =  load_img('/docker/etri/data/FlowerLF/IMG_2221_eslf.png')
        center_view_a = np.expand_dims(center_view_a, axis=0)
        center_view_b = np.expand_dims(center_view_b, axis=0)








        
        inLF_a = np.expand_dims(inLF_a, axis=0)
        inLF_b = np.expand_dims(inLF_b, axis=0)
        center_view = np.concatenate([center_view_a, center_view_b], axis=0)
        inLF = np.concatenate([inLF_a, inLF_b], axis=0)

        top[0].data[...] = center_view_a
        top[1].data[...] = inLF_a
        #print(top[0].data[...][0, 70:75, 110:115, 0])
        #print(top[1].data[...][0, 70:75, 110:115, 0])

    def backward(self, top, propagate_down, bottom):
        pass