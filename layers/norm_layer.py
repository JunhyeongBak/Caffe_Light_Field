import caffe
import numpy as np

class NormLayer(caffe.Layer):
    def setup(self, bottom, top):
        if len(bottom) != 1:
            raise Exception("Input size error.")
        
        params = eval(self.param_str)
        self.src_type = params['src_type']
        self.dst_type = params['dst_type']
       
    def reshape(self, bottom, top):
        self.src = np.cast['float32'](bottom[0].data)
        top[0].reshape(*self.src.shape)

    def forward(self, bottom, top):
        if self.src_type == 'uess': # [0, 255]
            if self.dst_type == 'udec': # [0, 1]
                dst = self.src / 255.0
            elif self.dst_type == 'dec': # [-1, 1]
                dst = ((self.src / 255.0) * 2.0) - 1.0
            else:
                raise Exception("Input type error1.")
        elif self.src_type == 'udec': # [0, 1]
            if self.dst_type == 'uess': # [0, 255]
                dst = self.src * 255.0
            elif self.dst_type == 'dec': # [-1, 1]
                dst = (self.src * 2.0) - 1.0
            else:
                raise Exception("Input type error2.")
        elif self.src_type == 'dec': # [-1, 1]
            if self.dst_type == 'uess': # [0, 255]
                dst = ((self.src / 2.0) + 0.5) * 255.0
            elif self.dst_type == 'udec': # [0, 1]
                dst = (self.src / 2.0) + 0.5
            else:
                raise Exception("Input type error3.")
        else:
            raise Exception("Input type error0.")

        top[0].data[...] = dst

    def backward(self, bottom, top):
        pass