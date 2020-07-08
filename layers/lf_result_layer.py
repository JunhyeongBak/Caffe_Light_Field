import caffe
import numpy as np
import cv2
import imageio
import warnings

warnings.filterwarnings(action='once')

class LfResultLayer(caffe.Layer):
    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        top[0].reshape(1)

    def forward(self, bottom, top):
        lf_flow_h = bottom[0].data[...]
        print(np.mean(lf_flow_h))
        lf_flow_v = bottom[1].data[...]
        print(np.mean(lf_flow_v))
        lf_input = bottom[2].data[...]
        lf_result = bottom[3].data[...]
        lf_label = bottom[4].data[...]
        
        cv2.imwrite('/docker/lf_depth/datas/hor_flow.png', -lf_flow_h[0, 0, :, :]*20)
        cv2.imwrite('/docker/lf_depth/datas/ver_flow.png', -lf_flow_v[0, 0, :, :]*20)
        cv2.imwrite('/docker/lf_depth/datas/input_flow.png', lf_input[0, 0, :, :])
        cv2.imwrite('/docker/lf_depth/datas/result_flow.png', lf_result[0, 0, :, :])
        cv2.imwrite('/docker/lf_depth/datas/label_flow.png', lf_label[0, 0, :, :])
        #print(lf_predict.shape)
        #imageio.mimsave('/docker/etri/data/FlowerLF/predict.gif', lf_predict[0, :, :, :], duration=0.05)
        #imageio.mimsave('/docker/etri/data/FlowerLF/label.gif', lf_label[0, :, :, :], duration=0.05)

        top[0].data[...] = 0

    def backward(self, top, propagate_down, bottom):
        pass