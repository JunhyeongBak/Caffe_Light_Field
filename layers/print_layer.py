import caffe
import numpy as np
import cv2

class PrintLayer(caffe.Layer):
    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.path = params['path']
        self.name = params['name']
        self.mult = params['mult']    
    
    def reshape(self, bottom, top):
        top[0].reshape(1)    
     
    def forward(self, bottom, top):
        im_org = bottom[0].data[0, :, :, :]*self.mult
        im_color = np.zeros((256*5, 256*5, 3))

        if bottom[0].data[...].shape[1] == 1:
            im_color[:, :, 0] = im_org[0, :, :]
            im_color[:, :, 1] = im_org[0, :, :]
            im_color[:, :, 2] = im_org[0, :, :]
            im_color = np.clip(im_color, 0, 255)
            im_color = im_color.astype('uint8')
            full_path = self.path+'/'+self.name+'.png'
            cv2.imwrite(full_path, im_color)
        else:
            for i in range(3):
                im_color[:, :, i] = im_org[i, :, :]
            im_color = np.clip(im_color, 0, 255)
            im_color = im_color.astype('uint8')
            full_path = self.path+'/'+self.name+'.png'
            cv2.imwrite(full_path, im_color)
        '''
        for b in range(bottom[0].data.shape[0]):
            for c in range(bottom[0].data.shape[1]):
                full_name = 'b'+str(b)+'_'+'c'+str(c)+'_'+self.name
                full_path = self.path+'/'+full_name+'.png'
                #print('<'+full_name+'>')
                #print(np.mean(bottom[0].data[b, c, :, :]))

                if self.mult < 256:
                    im_color = abs(bottom[0].data[b, c, :, :]*self.mult)
                    im_color = im_color.astype('uint8')
                    im_color = cv2.applyColorMap(im_color, cv2.COLORMAP_WINTER)
                    cv2.imwrite(full_path, im_color)
                else:
                    im_color = abs(bottom[0].data[b, c, :, :]*self.mult)
                    im_color = im_color.astype('uint8')
                    #im_color = cv2.applyColorMap(im_color, cv2.COLORMAP_WINTER)
                    cv2.imwrite(full_path, im_color)
        '''

    def backward(self, top, propagate_down, bottom):
        pass