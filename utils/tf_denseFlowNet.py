import numpy as np
import tensorflow as tf
import os
import sys
import glob
from random import shuffle
import cv2
from skimage import color
from bilinear_sampler import bilinear_sampler
from stn import spatial_transformer_network as transformer
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import math_ops

#Image path
TRAIN_PATH = 'LF_Flowers_Dataset/Flowers_8bit_train/*.png'
TEST_PATH = 'LF_Flowers_Dataset/Flowers_8bit_test/*.png'

#LF Parameter
ANGULAR_RES_X = 14
ANGULAR_RES_Y = 14
ANGULAR_RES_TARGET = 8*8
IMG_WIDTH = 3584
IMG_HEIGHT = 2688
SPATIAL_HEIGHT = int(IMG_HEIGHT / ANGULAR_RES_Y)
SPATIAL_WIDTH = int(IMG_WIDTH / ANGULAR_RES_X)
CH_INPUT = 3
CH_OUTPUT = 3
SHIFT_VALUE = 0.8

#Training parameter
BATCH_SIZE = 1
TRAIN_SIZE = 1.0
LR_G =  0.001 # 0.001 0.0005 0.00146 0.0002
EPOCH = 50000 # 150000
DECAY_STEP = 5000
DECAY_RATE = 0.90
LAMBDA_L1 = 10.0
LAMBDA_MV = 10.0 # 10.0
LAMBDA_TV = 1e-1 # 1e-1
EPS = 1e-6

#LF INDEX
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

####################################################################################################
#                                            UTILITES                                              #
####################################################################################################
def tf_image_translate(images, tx, ty, interpolation='BILINEAR'): 
# +tx -> shift left / +ty -> shift up 
    transforms = [1, 0, tx, 0, 1, ty, 0, 0]
    return tf.contrib.image.transform(images, transforms, interpolation)

def preprocess(image):
# Change pixel value range [0, 1] to [-1, 1]
    with tf.name_scope("preprocess"):
        return image * 2 - 1

def deprocess(image):
# Change pixel value range [-1, 1] to [0, 1]
    with tf.name_scope("deprocess"):
        return (image + 1) / 2

def process_LF(lf): 
# raw LF image to center LF image, 8x8 grid LF image, stacked LF image in channel axis(for EPI?)
    full_LF_crop = np.zeros((SPATIAL_HEIGHT, SPATIAL_WIDTH, 3, ANGULAR_RES_Y, ANGULAR_RES_X))
    
    for ax in range(ANGULAR_RES_X):
        for ay in range(ANGULAR_RES_Y):
            resized = lf[ay::ANGULAR_RES_Y, ax::ANGULAR_RES_X, :]
            resized2 = cv2.resize(resized, dsize=(SPATIAL_WIDTH, SPATIAL_HEIGHT), interpolation=cv2.INTER_LINEAR)
            full_LF_crop[:, :, :, ay, ax] = resized2 # prevent wrong spatial size
            
    # Take 8x8 LF on the middle, since the side part suffer from vignetting
    middle_LF = full_LF_crop[:, :, :, 3:11, 3:11] # Take 8x8 LF in 5D
    
    # To visualize the 8x8 LF
    for ax in range(8):
        for ay in range(8):
            if ay == 0:
                y_img = middle_LF[:,:,:,ay,ax]
            else:
                y_img = np.concatenate((y_img, middle_LF[:,:,:,ay,ax]), 0)
            if ax == 0 and ay ==0:
                LF_stack = middle_LF[:,:,:,0,0]
            else:
                LF_stack = np.concatenate((LF_stack, middle_LF[:,:,:,ay,ax]), 2)
        if ax == 0:
            LF_grid = y_img
        else:
            LF_grid = np.concatenate((LF_grid, y_img), 1)
        y_img = middle_LF[:,:,:,ay,ax]
    
    center_view = middle_LF[:,:,:,3,3]
    return center_view, LF_stack, LF_grid

####################################################################################################
#                                 PREPARE DATA & DATASET RECORD                                    #
####################################################################################################
def load_image(addr):
    img = cv2.imread(addr)
    
    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    center_view, GT, grid = process_LF(img)
    center_view = np.uint8(center_view)
    grid = np.uint8(grid)
    GT = np.uint8(GT)

    return center_view, GT, grid

def _bytes_feature(value):
# Change value to tf.train.Feature type
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def createDataRecord(out_filename, addrs):
# Record image data as tfrecord type

    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(out_filename)
    
    # print how many images are loaded every images
    for i in range(len(addrs)):
        if not i % 10:
            print('Train data: {}/{} images'.format(i, len(addrs)))
            sys.stdout.flush() # print() 명령어가 모였다가 한 번에 출력되는걸 방지하기 위해서 쓰는 코드임
        # Load the image
        img, label, grid = load_image(addrs[i]) 
        if img is None:
            continue
        if label is None:
            continue
        if grid is None:
            continue
        # Create a feature
        feature = {
            'image_raw': _bytes_feature(img.tostring()),
            'label': _bytes_feature(label.tostring()),
            'grid': _bytes_feature(grid.tostring())
        }
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
        
    writer.close()
    sys.stdout.flush()

####################################################################################################
#                                          INPUT PIPELINE                                          #
####################################################################################################
with tf.name_scope('Input_Pipeline'):
    gamma_val = tf.random_uniform(shape=[], minval=0.4, maxval=1.0)

    # Input center image (batch, height, width, ch)
    tf_x = tf.placeholder(tf.float32, [None, SPATIAL_HEIGHT, SPATIAL_WIDTH, CH_INPUT], name='Input')
    tf_x = tf.image.adjust_gamma(tf_x*255, gamma_val)
    image = tf.reshape(tf_x, [-1, SPATIAL_HEIGHT, SPATIAL_WIDTH, CH_INPUT], name='img_x')
    image_min = preprocess(image)
    
    # Input GT LF grid image for visualization (batch, height*8, width*8, ch) - 불필요 해
    tf_grid = tf.placeholder(tf.float32, [None, SPATIAL_HEIGHT*8, SPATIAL_WIDTH*8, CH_OUTPUT], name='Grids')
    tf_grid = tf.image.adjust_gamma(tf_grid*255, gamma_val)
    
    # Input stacked SAI (batch, height, width, ch*8*8)
    tf_y = tf.placeholder(tf.float32, [None, SPATIAL_HEIGHT, SPATIAL_WIDTH, CH_OUTPUT*64], name='Target')
    tf_y = tf.image.adjust_gamma(tf_y*255, gamma_val)
    color_norm = tf.reshape(tf_y, [-1, SPATIAL_HEIGHT, SPATIAL_WIDTH, CH_OUTPUT*64], name='img_y')
    color_norm_min = preprocess(color_norm)

####################################################################################################
#                                              NETWORK                                             #
####################################################################################################
def shift_value(i):
    if i<=7:
        tx = 3*SHIFT_VALUE
    elif i>7 and i<=15:
        tx = 2*SHIFT_VALUE
    elif i>15 and i<=23:
        tx = 1*SHIFT_VALUE
    elif i>23 and i<=31:
        tx = 0
    elif i>31 and i<=39:
        tx = -1*SHIFT_VALUE
    elif i>39 and i<=47:
        tx = -2*SHIFT_VALUE
    elif i>47 and i<=55:
        tx = -3*SHIFT_VALUE
    else:
        tx = -4*SHIFT_VALUE
    
    if i==0 or (i%8==0 and i>7):
        ty = 3*SHIFT_VALUE
    elif i == 1 or (i-1)%8==0:
        ty = 2*SHIFT_VALUE
    elif i == 2 or (i-2)%8==0:
        ty = 1*SHIFT_VALUE
    elif i == 3 or (i-3)%8==0:
        ty = 0
    elif i == 4 or (i-4)%8==0:
        ty = -1*SHIFT_VALUE
    elif i == 5 or (i-5)%8==0:
        ty = -2*SHIFT_VALUE
    elif i == 6 or (i-6)%8==0:
        ty = -3*SHIFT_VALUE
    else:
        ty = -4*SHIFT_VALUE
        
    return tx, ty

def add_layer(input_=None, rate=1):
    c = tf.nn.relu(input_)
    c = tf.layers.conv2d(c, 12, 3, padding='SAME', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal(), dilation_rate=rate)
    return tf.concat([input_, c], -1)

def transition(input_=None):
    shape = input_.get_shape().as_list()
    filters = shape[-1] # -가 붙었다는 가장 뒤에 것을 읽는다는 것 임, 그러므로 아마도 50, 86, 122, 158 순으로 진행될 듯
    c = tf.nn.relu(input_)
    c = tf.layers.conv2d(c, filters, 3, padding='SAME', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
    # No average pooling
    return c

def flow_layer(input_=None):
    shape = input_.get_shape().as_list()
    filters = shape[-1]
    c = tf.nn.relu(input_)
    c = tf.layers.conv2d(c, filters, 3, padding='SAME', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
    c = tf.layers.conv2d(c, (ANGULAR_RES_TARGET)*2, 3, padding='SAME', activation=None, kernel_initializer=tf.keras.initializers.he_normal())
    return c

def LF_Synthesis(input_yuv=None):
    input_ = input_yuv[:,:,:,0:1]
    with tf.name_scope('Initial_Conv'):
        conv = tf.layers.conv2d(input_yuv, 14, 3, padding='SAME', activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal(), name='INITIAL_CONV_1')
        
    with tf.name_scope('Flow_Generator'):
        with tf.name_scope('Block_1'):
            for i in range(3):
                conv = add_layer(conv, 2)
            conv = transition(conv)

        with tf.name_scope('Block_2'):
            for i in range(3):
                conv = add_layer(conv, 4)
            conv = transition(conv)

        with tf.name_scope('Block_3'):
            for i in range(3):
                conv = add_layer(conv, 8)
            conv = transition(conv)

        with tf.name_scope('Block_4'):
            for i in range(3):
                conv = add_layer(conv,16)
            conv = transition(conv)

        with tf.name_scope('Flow'):
            flow_LF = flow_layer(conv)

    with tf.name_scope('Estimation_Layer'):
        y = input_
        
        for i in range(ANGULAR_RES_TARGET): 
            tx, ty = shift_value(i)
            image_shift = tf_image_translate(image_min, tx, ty)
            y_shift = tf_image_translate(y, tx, ty)
            
            if i==0:
                pred_LF = bilinear_sampler(image_shift, flow_LF[:, :, :, i*2:(i*2)+2])
                pred_LF_loss = bilinear_sampler(y_shift, flow_LF[:, :, :, i*2:(i*2)+2])
            elif i==27: # Center image
                pred_LF = tf.concat((pred_LF, image_min), -1)
                pred_LF_loss = tf.concat((pred_LF_loss, y), -1)
            else:
                trans_image = bilinear_sampler(image_shift, flow_LF[:, :, :, i*2:(i*2)+2])
                pred_LF = tf.concat((pred_LF, trans_image), -1)
                trans_image = bilinear_sampler(y_shift, flow_LF[:, :, :, i*2:(i*2)+2])
                pred_LF_loss = tf.concat((pred_LF_loss, trans_image), -1)  

        print('pred_LF', pred_LF.shape)           
        print('pred_LF_loss', pred_LF_loss.shape)
        print('flow_LF', flow_LF.shape)  

    return pred_LF, pred_LF_loss, flow_LF

with tf.name_scope('View_Synthesis'):
    with tf.name_scope('Main_Network'):
        yuv = tf.image.rgb_to_yuv(image)
        y = preprocess(yuv[:,:,:,0:3])
        pred_LF, pred_LF_loss, flow_LF = LF_Synthesis(y)
        pred_LF_norm = deprocess(pred_LF)
    
    with tf.name_scope('EPI_Slicing'):
        for i in range(ANGULAR_RES_TARGET):  
            temp = tf.image.rgb_to_yuv(color_norm[:,:,:,i*3:(i*3)+3])
            if i == 0:
                y_GT = temp[:,:,:,0:1]
            else:
                y_GT = tf.concat([y_GT, temp[:,:,:,0:1]], -1)
        y_GT = preprocess(y_GT)    
        
        center_width = int(SPATIAL_WIDTH/2) # 중간지점 까지의 폭
        center_height = int(SPATIAL_HEIGHT/2) # 중간지점 까지의 높이

        slice_epi_H = pred_LF_loss[:, center_height:center_height+1, :, :]
        slice_epi_GT_H = y_GT[:, center_height:center_height+1, :, :]
        slice_epi_V = pred_LF_loss[:, :, center_width:center_width+1, :]
        slice_epi_GT_V = y_GT[:, :, center_width:center_width+1, :]
        print('slice_epi_H : ', slice_epi_H)
        print('slice_epi_GT_H : ', slice_epi_GT_H)
        print('slice_epi_V : ', slice_epi_V)
        print('slice_epi_GT_V : ', slice_epi_GT_V)
        # 포개어진 상태로 중간지점만 전부 추출하면 EPI를 구할 수 있고 이걸로 loss를 구함(predict와 ground truth모두 구해야함)
        
        # EPI를 사용자가 보려면 포개어 놓은게 아니라 펴서 붙여놓아야하는데
        # 수평 EPI는 SPATIAL_HEIGHT(=SPATIAL_WIDTH)가 앞쪽이라 reshape만으로 그게 가능함
        # 수직 EPI는 그게 불가능해서 일일이 펴서 붙여줘야함
        # 분석하면서 알게된 사실은 지금 여기서 만든 볼 수 있는 EPI는 흑백이고 이를 실제로 볼 수 있게 만드는 부분이 없는 쓰레기 코드임
        for j in range(8):
            for i in range(8):
                temp = slice_epi_H[:, :, :, (i*8)+(j):(i*8)+(j)+1]
                temp2 = slice_epi_GT_H[:, :, :, (i*8)+(j):(i*8)+(j)+1]
                temp3 = pred_LF_loss[:, :, :, (i*8)+(j):(i*8)+(j)+1]
                temp4 = y_GT[:, :, :, (i*8)+(j):(i*8)+(j)+1]
                
                if i==0 and j==0:
                    epi_synth_H = temp
                    epi_GT_H = temp2
                    pred_LF_loss_H = temp3
                    y_GT_H = temp4
                else:
                    epi_synth_H = tf.concat([epi_synth_H,temp], 1)
                    epi_GT_H = tf.concat([epi_GT_H,temp2], 1)
                    pred_LF_loss_H = tf.concat([pred_LF_loss_H,temp3], -1)
                    y_GT_H = tf.concat([y_GT_H,temp4], -1)
        
        # pred_LF_loss_H = pred_LF_loss
        # y_GT_H = y_GT
        print('pred_LF_loss_H : ', pred_LF_loss_H.shape)
        print('y_GT_H : ', y_GT_H.shape)
        
        epi_synth_V = tf.reshape(slice_epi_V, [-1, SPATIAL_HEIGHT, 64, 1])
        epi_GT_V = tf.reshape(slice_epi_GT_V, [-1, SPATIAL_HEIGHT, 64, 1])

####################################################################################################
#                                        MODEL SUMMARY                                             #
####################################################################################################
def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)
    
model_summary()

####################################################################################################
#                                               LOSS                                               #
####################################################################################################
def total_variation_self(images, name=None):
    ndims = images.get_shape().ndims

    if ndims == 3:    
        pixel_dif1 = (images[1:, :, :] - images[:-1, :, :])
        pixel_dif2 = (images[:, 1:, :] - images[:, :-1, :])
        pixel_dif1*=pixel_dif1
        pixel_dif2*=pixel_dif2
        # Sum for all axis. (None is an alias for all axis.)
        sum_axis = None
    elif ndims == 4:
        pixel_dif1 = (images[:, 1:, :, :] - images[:, :-1, :, :])
        pixel_dif2 = (images[:, :, 1:, :] - images[:, :, :-1, :])
        pixel_dif1*=pixel_dif1
        pixel_dif2*=pixel_dif2
        # Only sum for the last 3 axis.
        # This results in a 1-D tensor with the total variation for each image.
        sum_axis = [1, 2, 3]

    tot_var = (math_ops.reduce_mean(math_ops.abs(pixel_dif1), axis=sum_axis) +
                math_ops.reduce_mean(math_ops.abs(pixel_dif2), axis=sum_axis))

    return tot_var

with tf.name_scope('Pixel_Based_Loss'):
    #EPI based loss in horizontal and vertical direction sampled every 8 angular images
    with tf.name_scope('L_Loss'):
        variance_loss = mean_loss = pixel_loss_V = pixel_loss_H = edge_loss_V = edge_loss_H = 0

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

    with tf.name_scope('EG_Loss'):

        y_GT_edgew
        for i in range(64):
            if i == 0:
                temp1_edge = tf.image.sobel_edges(tf.expand_dims(y_GT[:,:,:,i], 1))
                y_GT_edge = (temp1_edge[:, :, :, :, 0] + temp1_edge[:, :, :, :, 1]) / 2.
                temp2_edge = tf.image.sobel_edges(tf.expand_dims(pred_LF_loss[:,:,:,i], 1))
                pred_LF_loss_edge[:,:,:,i] = (temp2_edge[:, :, :, :, 0] + temp2_edge[:, :, :, :, 1]) / 2.
                temp3_edge = tf.image.sobel_edges(tf.expand_dims(y_GT_H[:,:,:,i], 1))
                y_GT_H_edge[:,:,:,i] = (temp3_edge[:, :, :, :, 0] + temp3_edge[:, :, :, :, 1]) / 2.
                temp4_edge = tf.image.sobel_edges(tf.expand_dims(pred_LF_loss_H[:,:,:,i], 1))
                pred_LF_loss_H_edge[:,:,:,i] = (temp4_edge[:, :, :, :, 0] + temp4_edge[:, :, :, :, 1]) / 2.

        for i in range(8):  
            temp1 = tf.reduce_mean(y_GT_edge[:,:,:,(i*8):(i*8)+8], axis=-1)
            temp2 = tf.reduce_mean(pred_LF_loss_edge[:,:,:,(i*8):(i*8)+8], axis=-1)
            edge_loss_V += tf.losses.absolute_difference(temp1, temp2)
            
            temp3 = tf.reduce_mean(y_GT_H_edge[:,:,:,(i*8):(i*8)+8], axis=-1)
            temp4 = tf.reduce_mean(pred_LF_loss_H_edge[:,:,:,(i*8):(i*8)+8], axis=-1)
            edge_loss_H += tf.losses.absolute_difference(temp3, temp4)

        edge_loss_V += tf.losses.absolute_difference(tf.reduce_mean(y_GT_edge[:,:,:,:], axis=-1), tf.reduce_mean(pred_LF_loss_edge[:,:,:,:], axis=-1))
        edge_loss_H += tf.losses.absolute_difference(temp3, temp4)

        tf.summary.scalar('edge_loss_V', edge_loss_V)
        tf.summary.scalar('edge_loss_H', edge_loss_H)        
 
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
    
    
    #Reshape the output into a grid LF
    for i in range(1,ANGULAR_RES_TARGET):
        if i==8:
            grid_LF = y_img
            y_img = pred_LF[:,:,:,i*3:(i*3)+3]
                
        elif i%8==0 and i>8:
            grid_LF = tf.concat([grid_LF, y_img], 2)
            y_img = pred_LF[:,:,:,i*3:(i*3)+3]

        elif i == 63:
            y_img = tf.concat([y_img, pred_LF[:,:,:,i*3:(i*3)+3]], 1)
            grid_LF = tf.concat([grid_LF, y_img], 2)

        else:
            y_img = tf.concat([y_img, pred_LF[:,:,:,i*3:(i*3)+3]], 1)

with tf.name_scope('Evaluation'):
    psnr = tf.image.psnr(color_norm, pred_LF_norm, max_val=1.0)
    tf.summary.scalar('PSNR', psnr[0])
    
    ssim = tf.image.ssim(color_norm, pred_LF_norm, max_val=1.0)
    tf.summary.scalar('SSIM', ssim[0])

####################################################################################################
#                                             TRAIN OP                                             #
####################################################################################################
with tf.name_scope('Train'):
    global_step_G = tf.Variable(0, dtype=tf.float32)
    learning_rate_G = tf.train.exponential_decay(
                      LR_G,                  # Base learning rate.
                      global_step_G,         # Current index into the dataset.
                      DECAY_STEP,            # Decay step.
                      DECAY_RATE,            # Decay rate.
                      staircase=True)
    tf.summary.scalar('LR_G', learning_rate_G)
    train_G = tf.train.AdamOptimizer(learning_rate=learning_rate_G, name='optimizer_G').minimize(Total_Loss, global_step=global_step_G)

####################################################################################################
#                                          VISUALIZATION                                           #
####################################################################################################
with tf.name_scope('Visualization'):     
    #Open the predicted LF into grid for EPI visualization
    grid_LF = tf.zeros_like(image)
    y_img = pred_LF[:,:,:,0:3]
    
    for i in range(1,ANGULAR_RES_TARGET):
        if i==8:
            grid_LF = y_img
            y_img = pred_LF[:,:,:,i*3:(i*3)+3]
        elif i%8==0 and i>8:
            grid_LF = tf.concat([grid_LF, y_img], 2)
            y_img = pred_LF[:,:,:,i*3:(i*3)+3]
        elif i == 63:
            y_img = tf.concat([y_img, pred_LF[:,:,:,i*3:(i*3)+3]], 1)
            grid_LF = tf.concat([grid_LF, y_img], 2)
        else:
            y_img = tf.concat([y_img, pred_LF[:,:,:,i*3:(i*3)+3]], 1)

    #TF Summary image
    grid_LF_show = deprocess(grid_LF)
    output_image = tf.summary.image('Synthesized_LF', tf.cast(tf.reshape(grid_LF_show*255, 
                            [-1, SPATIAL_HEIGHT*8, SPATIAL_WIDTH*8, CH_OUTPUT]), tf.uint8) , 1)
    
    #EPI VISUALIZATION
    with tf.name_scope('EPI'): 
        epi_synth_H = deprocess(epi_synth_H)
        epi_GT_H = deprocess(epi_GT_H)
        epi_synth_V = deprocess(epi_synth_V)
        epi_GT_V = deprocess(epi_GT_V)
        epi_horizontal = tf.summary.image('epi_horizontal', tf.cast(tf.reshape(epi_synth_H*255, 
                                [-1, 64, SPATIAL_WIDTH, 1]), tf.uint8) , 1)
        epi_horizontal_GT = tf.summary.image('epi_horizontal_GT', tf.cast(tf.reshape(epi_GT_H*255, 
                                [-1, 64, SPATIAL_WIDTH, 1]), tf.uint8) , 1)
        epi_vertical = tf.summary.image('epi_vertical', tf.cast(tf.reshape(epi_synth_V*255, 
                                [-1, SPATIAL_HEIGHT, 64, 1]), tf.uint8) , 1)
        epi_vertical_GT = tf.summary.image('epi_vertical_GT', tf.cast(tf.reshape(epi_GT_V*255, 
                                [-1, SPATIAL_HEIGHT, 64, 1]), tf.uint8) , 1)  
        
with tf.name_scope('Mean_Variance'):         
        mean_img = tf.summary.image('mean_GT', tf.cast(tf.reshape(mean*255, 
                            [-1, SPATIAL_HEIGHT, SPATIAL_WIDTH, 1]), tf.uint8) , 1)
        var_img = tf.summary.image('var_GT', tf.cast(tf.reshape(variance*1000, 
                            [-1, SPATIAL_HEIGHT, SPATIAL_WIDTH, 1]), tf.uint8) , 1)
        
        mean_img2 = tf.summary.image('mean_output', tf.cast(tf.reshape(mean2*255, 
                            [-1, SPATIAL_HEIGHT, SPATIAL_WIDTH, 1]), tf.uint8) , 1)
        var_img2 = tf.summary.image('var_output', tf.cast(tf.reshape(variance2*1000, 
                            [-1, SPATIAL_HEIGHT, SPATIAL_WIDTH, 1]), tf.uint8) , 1)

"""
####################################################################################################
#                                            VALIDATION                                            #
####################################################################################################
#SLICE another EPI in the LF to make sure not only cross EPI is correct but all EPI is correct
with tf.name_scope('VALIDATION'): 
    center_width = tf.random_uniform(shape=[], minval=10, maxval=SPATIAL_WIDTH-10, dtype=tf.int32)
    center_height = tf.random_uniform(shape=[], minval=10, maxval=SPATIAL_HEIGHT-10, dtype=tf.int32)

    slice_epi_H2 = pred_LF_norm[:, center_height:center_height+1, :, :]
    slice_epi_V2 = pred_LF_norm[:, :, center_width:center_width+1, :]
    
    slice_epi_GT_H2 = color_norm[:, center_height:center_height+1, :, :]
    slice_epi_GT_V2 = color_norm[:, :, center_width:center_width+1, :]
        
    #Because of the stack is in row order for horizontal EPI it cannot be directly reshaped
    for j in range(8):
        for i in range(8):
            temp = slice_epi_H2[:, :, :, (i*8*3)+(j*3):(i*8*3)+(j*3)+3]
            temp2 = slice_epi_GT_H2[:, :, :, (i*8*3)+(j*3):(i*8*3)+(j*3)+3]
            if i==0 and j==0:
                epi_synth_H2 = temp
                epi_GT_H2 = temp2
            else:
                epi_synth_H2 = tf.concat([epi_synth_H2,temp], 1)
                epi_GT_H2 = tf.concat([epi_GT_H2,temp2], 1)

    epi_synth_V2 = tf.reshape(slice_epi_V2, [-1,SPATIAL_HEIGHT, 64, 3])
    epi_GT_V2 = tf.reshape(slice_epi_GT_V2, [-1,SPATIAL_HEIGHT, 64, 3])

    EPI_H2 = tf.summary.image('epi_horizontal2', tf.cast(tf.reshape(epi_synth_H2*255, 
                            [-1, 64, SPATIAL_WIDTH, 3]), tf.uint8) , 1)
    epi_horizontal_GT2 = tf.summary.image('epi_horizontal_GT2', tf.cast(tf.reshape(epi_GT_H2*255, 
                                [-1, 64, SPATIAL_WIDTH, 3]), tf.uint8) , 1)
    EPI_V2 = tf.summary.image('epi_vertical2', tf.cast(tf.reshape(epi_synth_V2*255, 
                            [-1, SPATIAL_HEIGHT, 64, 3]), tf.uint8) , 1)
    epi_vertical_GT2 = tf.summary.image('epi_vertical_GT2', tf.cast(tf.reshape(epi_GT_V2*255, 
                                [-1, SPATIAL_HEIGHT, 64, 3]), tf.uint8) , 1)
"""
####################################################################################################
#                                          INPUT PARSING                                           #
####################################################################################################
#To get one record and parse it to get the label and image out
def parser(record):
    keys_to_features = {
        "image_raw": tf.FixedLenFeature([], tf.string),
        "label":     tf.FixedLenFeature([], tf.string),
        "grid":     tf.FixedLenFeature([], tf.string)
    }
    #Read one record
    parsed = tf.parse_single_example(record, keys_to_features)
    #Take the image and bytes
    image = tf.decode_raw(parsed["image_raw"], tf.uint8)
    label = tf.decode_raw(parsed["label"], tf.uint8)
    grid = tf.decode_raw(parsed["grid"], tf.uint8)
    #Cast to float
    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.float32)
    grid = tf.cast(grid, tf.float32)
    
    image = tf.reshape(image, shape=[SPATIAL_HEIGHT, SPATIAL_WIDTH, CH_INPUT])
    label = tf.reshape(label, shape=[SPATIAL_HEIGHT, SPATIAL_WIDTH, CH_OUTPUT*64])
    grid = tf.reshape(grid, shape=[SPATIAL_HEIGHT*8, SPATIAL_WIDTH*8, CH_OUTPUT])
    #Normalize the input and label into [0...1]
    image = tf.divide(image, 255)
    label = tf.divide(label, 255)

    return {'image': image}, {'label': label}, {'grid': grid}

def input_fn(filenames):
    #Create data record
    dataset = tf.data.TFRecordDataset(filenames=filenames, num_parallel_reads=1)
    dataset = dataset.map(parser, num_parallel_calls=1)
    dataset = dataset.shuffle(40).repeat().batch(BATCH_SIZE)
    #dataset = dataset.prefetch(buffer_size=2)
    return dataset

def test_fn(filenames):
    #Create data record
    dataset = tf.data.TFRecordDataset(filenames=filenames, num_parallel_reads=1)
    dataset = dataset.map(parser, num_parallel_calls=1)
    dataset = dataset.batch(10)
    return dataset

def train_input_fn():
    return input_fn(filenames=["train.tfrecords"])

def test_input_fn():
    return test_fn(filenames=["test.tfrecords"])

####################################################################################################
#                                        CREATE TRAIN SET                                          #
####################################################################################################
with tf.name_scope('Create_Training_Set'):
    train_dataset = train_input_fn()
    iterator = train_dataset.make_initializable_iterator()
    next_batch = iterator.get_next()


####################################################################################################
#                                               TRAIN                                              #
####################################################################################################
merged = tf.summary.merge_all()
saver = tf.train.Saver()

config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)


sess.run(tf.group(tf.global_variables_initializer(), iterator.initializer))
writer = tf.summary.FileWriter('log/Synthesis/LF',sess.graph)

run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True, trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()

for step in range(EPOCH+1):
    train_x, train_y, train_grid = sess.run(next_batch)                       
    
    _, G_loss_, psnr_, ssim_ = sess.run([train_G, Total_Loss, psnr, ssim], 
    {tf_x:train_x['image'], tf_y:train_y['label'], tf_grid:train_grid['grid']})
   
    if step%350 == 0:
        #writer.add_run_metadata(run_metadata, 'step%d' % step)
        summary_ = sess.run(merged, {tf_x:train_x['image'], tf_y:train_y['label'], tf_grid:train_grid['grid']}
                           , options=run_options, run_metadata=run_metadata)
        writer.add_summary(summary_, step)     
        print('Step:', step, '| loss:%.5f' %G_loss_, '| PSNR:%.3f'  %psnr_[0], '| SSIM:%.3f' %ssim_[0])
        
    if step%30000==0:
        save_path = saver.save(sess, "saver/Synthesis/model%i.ckpt" %step)
        print("Model saved in path: %s" % save_path)

save_path = saver.save(sess, "saver/Synthesis/model%i.ckpt" %step)
print("Model saved in path: %s" % save_path)