import caffe
import numpy as np
import cv2

class WarpingLayer(caffe.Layer):
    def setup(self, bottom, top):
        pass
    
    def reshape(self, bottom, top):
        self.src_blob = bottom[0].data[...]
        self.flow_blob = bottom[1].data[...]
        top[0].reshape(*self.src_blob.shape)

    def forward(self, bottom, top):
        c = self.src_blob.shape[1]

        if c == 3:
            src = np.moveaxis(np.squeeze(self.src_blob, 0), 0, -1)
            flow = np.moveaxis(np.squeeze(self.flow_blob, 0), 0, -1)
        elif c == 1:
            src = np.squeeze(np.squeeze(self.src_blob, 0), 0)
            flow = np.moveaxis(np.squeeze(self.flow_blob, 0), 0, -1)
        else:
            raise Exception('This is not a normal image !!!')

        h, w = flow.shape[:2]
        flow[:,:,0] += np.arange(w)
        flow[:,:,1] += np.arange(h)[:,np.newaxis]

        dst = np.zeros((256, 256, 3))
        for i in range(3):
            dst[:, :, i] = cv2.remap(src[:, :, i], flow, None, interpolation = cv2.INTER_LINEAR, borderMode = cv2.BORDER_REPLICATE)

        if c == 3:
            dst_blob = np.expand_dims(np.moveaxis(dst, -1, 0), 0)
        elif c == 1:
            dst_blob = np.expand_dims(np.expand_dims(dst, 0), 0)
        else:
            raise Exception('This is not a normal image !!!')

        top[0].data[...] = dst_blob

    def backward(self, top, propagate_down, bottom):
        pass