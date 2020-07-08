import caffe
import numpy as np
import cv2

class BilinearSampler(caffe.Layer):

    def setup(self, bottom, top):
        # params = eval(self.param_str)
        #params = eval(self.param_str)
        #self.tx = params["tx"]
        #self.ty = params["ty"]
        pass

    def _get_grid_array(self, N, H, W, h, w):
        N_i = np.arange(N)
        H_i = np.arange(h+1, h+H+1)
        W_i = np.arange(w+1, w+W+1)
        n, h, w, = np.meshgrid(N_i, H_i, W_i, indexing='ij')
        n = np.expand_dims(n, axis=3) # [N, H, W, 1]
        h = np.expand_dims(h, axis=3) # [N, H, W, 1]
        w = np.expand_dims(w, axis=3) # [N, H, W, 1]
        n = np.cast[np.float32](n) # [N, H, W, 1]
        h = np.cast[np.float32](h) # [N, H, W, 1]
        w = np.cast[np.float32](w) # [N, H, W, 1]

        return n, h, w
        
    def reshape(self, bottom, top):
        ### blob to numpy image ###
        # blob's shape : (n, c, h, w)
        # images's shape : (n, h, w, c)
        # image's shape : (h, w, c)
        self.src = np.asarray(bottom[0].data[...])
        self.vec = np.asarray(bottom[1].data[...])
        self.dst = np.zeros((self.src.shape[0], self.src.shape[1], self.src.shape[2], self.src.shape[3]))
        top[0].reshape(*self.dst.shape)

    def forward(self, bottom, top):
        v = np.ones([self.shape[0], self.shape[2], self.shape[3], 2]) * 20.

        shape = self.src.shape
        N = shape[0]

        H_ = H = shape[1]
        W_ = W = shape[2]
        h = w = 0

        npad = ((0,0), (1,1), (1,1), (0,0))
        self.src = np.pad(self.src, npad, 'constant', constant_values=(0))

        vy, vx = np.split(v, 2, axis=3)

        n, h, w = self._get_grid_array(N, H, W, h, w)

        vx0 = np.floor(vx)
        vy0 = np.floor(vy)
        vx1 = vx0 + 1
        vy1 = vy0 + 1

        H_1 = np.cast[np.float32](H_+1)
        W_1 = np.cast[np.float32](W_+1)

        iy0 = np.clip(vy0 + h, 0., H_1)
        iy1 = np.clip(vy1 + h, 0., H_1)
        ix0 = np.clip(vx0 + w, 0., W_1)
        ix1 = np.clip(vx1 + w, 0., W_1)

        i00 = np.concatenate([n, iy0, ix0], axis=3)
        i01 = np.concatenate([n, iy1, ix0], axis=3)
        i10 = np.concatenate([n, iy0, ix1], axis=3)
        i11 = np.concatenate([n, iy1, ix1], axis=3)

        i00 = np.cast[np.int32](i00)
        i01 = np.cast[np.int32](i01)
        i10 = np.cast[np.int32](i10)
        i11 = np.cast[np.int32](i11)

        idx_shpae = i00.shape
        idx_long = idx_shpae[0] * idx_shpae[1] * idx_shpae[2]

        x00 = []
        i00 = i00.reshape(-1, 3)
        i00_tup = [tuple(x) for x in i00.tolist()]
        for i in range(0, idx_long):
            x00.append(self.src[i00_tup[i]].tolist())
        x00 = np.asarray(x00)
        x00 = x00.reshape(idx_shpae[0], idx_shpae[1], idx_shpae[2], -1)

        x01 = []
        i01 = i01.reshape(-1, 3)
        i01_tup = [tuple(x) for x in i01.tolist()]
        for i in range(0, idx_long):
            x01.append(self.src[i01_tup[i]].tolist())
        x01 = np.asarray(x01)
        x01 = x01.reshape(idx_shpae[0], idx_shpae[1], idx_shpae[2], -1)

        x10 = []
        i10 = i10.reshape(-1, 3)
        i10_tup = [tuple(x) for x in i10.tolist()]
        for i in range(0, idx_long):
            x10.append(self.src[i10_tup[i]].tolist())
        x10 = np.asarray(x10)
        x10 = x10.reshape(idx_shpae[0], idx_shpae[1], idx_shpae[2], -1)

        x11 = []
        i11 = i11.reshape(-1, 3)
        i11_tup = [tuple(x) for x in i11.tolist()]
        for i in range(0, idx_long):
            x11.append(self.src[i11_tup[i]].tolist())
        x11 = np.asarray(x11)
        x11 = x11.reshape(idx_shpae[0], idx_shpae[1], idx_shpae[2], -1)

        w00 = np.cast[np.float32]((vx1 - vx) * (vy1 - vy))
        w01 = np.cast[np.float32]((vx1 - vx) * (vy - vy0))
        w10 = np.cast[np.float32]((vx - vx0) * (vy1 - vy))
        w11 = np.cast[np.float32]((vx - vx0) * (vy - vy0))
        output1 = np.add(w00*x00, w01*x01)
        output2 = np.add(w10*x10, w11*x11)
        output3 = np.add(output1, output2)

        self.dst = output3
        top[0].data[...] = self.dst

    def backward(self, bottom, top):
        pass