import caffe
import numpy as np

class LfRelocLayer(caffe.Layer):
    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        self.lf_shape = bottom[0].data[...].shape
        top[0].reshape(*self.lf_shape)
        top[1].reshape(*self.lf_shape)

    def forward(self, bottom, top):
        pred_LF_loss_H = np.asarray([])
        y_GT_H = np.asarray([])
        for j in range(8):
            for i in range(8):
                temp3 = bottom[0].data[...][:, (i*8)+(j):(i*8)+(j)+1, :, :]
                temp4 = bottom[1].data[...][:, (i*8)+(j):(i*8)+(j)+1, :, :]

                if i==0 and j==0:
                    pred_LF_loss_H = temp3
                    y_GT_H = temp4
                else:
                    pred_LF_loss_H = np.concatenate([pred_LF_loss_H,temp3], axis=1)
                    y_GT_H = np.concatenate([y_GT_H,temp4], axis=1)
        
        top[0].data[...] = pred_LF_loss_H
        top[1].data[...] = y_GT_H

    def backward(self, top, propagate_down, bottom):
        pass