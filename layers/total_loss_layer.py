import caffe
import numpy as np
import cv2

class TotalLossLayer(caffe.Layer):
    def setup(self, bottom, top):
        #params = eval(self.param_str)
        #self.shift_val = params['shift_val']
        pass

    def reshape(self, bottom, top):
        self.src = np.asarray(bottom[0].data[...])
        self.gt = np.asarray(bottom[1].data[...])
        self.loss = 0
        top[0].reshape(1)

    def absolute_difference(self, labels, predictions, weights=1.0, reduction='mean'):
        if reduction == 'mean':
            reduction_fn = np.mean
        elif reduction == 'sum':
            reduction_fn = np.sum
        else:
            # You could add more reductions
            pass
        labels = np.cast(labels, np.float32)
        predictions = np.cast(predictions, np.float32)
        losses = np.abs(np.subtract(predictions, labels))
        weights = np.cast(weights, np.float32)
        res = losses_utils.compute_weighted_loss(losses,
                                                weights,
                                                reduction=tf.keras.losses.Reduction.NONE)

        return reduction_fn(res, axis=None)

res = absolute_difference(labels, predictions)

    def forward(self, bottom, top):
        ### L_Loss ###
        variance_loss = mean_loss = pixel_loss_V = pixel_loss_H = 0

        for j in range(3):
            for i in range(3):
                temp1 = self.src[:, :, :, (i*3)+(j):(i*3)+(j)+1]
                temp2 = self.gt[:, :, :, (i*3)+(j):(i*3)+(j)+1]
                
                if i==0 and j==0:
                    self.src_h = temp1
                    self.gt_h = temp2
                else:
                    self.src_h = np.concatenate([self.src_h, temp1], axis=-1)
                    self.gt_h = np.concatenate([self.gt_h, temp2], axis=-1)

        for i in range(8):  
            temp3 = np.mean(self.gt[:,:,:,(i*3):(i*3)+3], axis=-1)
            temp4 = np.mean(self.src[:,:,:,(i*3):(i*3)+3], axis=-1)

            pixel_loss_V += tf.losses.absolute_difference(temp1, temp2)
            
            temp3 = tf.reduce_mean(y_GT_H[:,:,:,(i*8):(i*8)+8], axis=-1)
            temp4 = tf.reduce_mean(pred_LF_loss_H[:,:,:,(i*8):(i*8)+8], axis=-1)
            pixel_loss_H += tf.losses.absolute_difference(temp3, temp4)

        pixel_loss_V += tf.losses.absolute_difference(tf.reduce_mean(y_GT[:,:,:,:], axis=-1), tf.reduce_mean(pred_LF_loss[:,:,:,:], axis=-1))
        pixel_loss_H += tf.losses.absolute_difference(temp3, temp4)


    def backward(self, bottom, top):
        pass


with tf.name_scope('Pixel_Based_Loss'):
    #EPI based loss in horizontal and vertical direction sampled every 8 angular images
    with tf.name_scope('L_Loss'):
        variance_loss = mean_loss = pixel_loss_V = pixel_loss_H = 0

        for i in range(8):  
            temp1 = tf.reduce_mean(y_GT[:,:,:,(i*8):(i*8)+8], axis=-1)
            temp2 = tf.reduce_mean(pred_LF_loss[:,:,:,(i*8):(i*8)+8], axis=-1)
            pixel_loss_V += tf.losses.absolute_difference(temp1, temp2)
            
            temp3 = tf.reduce_mean(y_GT_H[:,:,:,(i*8):(i*8)+8], axis=-1)
            temp4 = tf.reduce_mean(pred_LF_loss_H[:,:,:,(i*8):(i*8)+8], axis=-1)
            pixel_loss_H += tf.losses.absolute_difference(temp3, temp4)

        pixel_loss_V += tf.losses.absolute_difference(tf.reduce_mean(y_GT[:,:,:,:], axis=-1), tf.reduce_mean(pred_LF_loss[:,:,:,:], axis=-1))
        pixel_loss_H += tf.losses.absolute_difference(temp3, temp4)

        tf.summary.scalar('pixel_loss_V', pixel_loss_V)
        tf.summary.scalar('pixel_loss_H', pixel_loss_H)        
 
    with tf.name_scope('MV_Loss'):  
        mean, variance = tf.nn.moments(deprocess(y_GT), -1)
        mean2, variance2 = tf.nn.moments(deprocess(pred_LF_loss), -1)
        
        mean = tf.where(tf.is_nan(mean), tf.zeros_like(mean), mean)
        mean2 = tf.where(tf.is_nan(mean2), tf.zeros_like(mean2), mean2)
        variance = tf.where(tf.is_nan(variance), tf.zeros_like(variance), variance)
        variance2 = tf.where(tf.is_nan(variance2), tf.zeros_like(variance2), variance2)
        
        variance = tf.sqrt(variance+EPS)
        variance2 = tf.sqrt(variance2+EPS)
        
        mean_loss = tf.losses.absolute_difference(mean, mean2)
        variance_loss = tf.losses.absolute_difference(variance, variance2)
        tf.summary.scalar('mean_loss', mean_loss)
        tf.summary.scalar('variance_loss', variance_loss)
        
    # Total variation loss for flow to surpress amount of artifact and smooth flow
    with tf.name_scope('Total_Variation_Loss'): 
        tv_loss_x = (total_variation_self(flow_LF[:,:,:,0::2]))
        tv_loss_y = (total_variation_self(flow_LF[:,:,:,1::2]))
        tv_loss = tf.reduce_mean(tv_loss_x) + tf.reduce_mean(tv_loss_y)
        tf.summary.scalar('TV_loss', tv_loss)
    
with tf.name_scope('Total_Loss'):
    Total_Loss = (LAMBDA_L1 * pixel_loss_V) + (LAMBDA_L1 * pixel_loss_H) \
                + (LAMBDA_TV * tv_loss) + (LAMBDA_MV*mean_loss) + (LAMBDA_MV*variance_loss)
    tf.summary.scalar('Total_Loss', Total_Loss)
    y_img = pred_LF[:,:,:,0:3]