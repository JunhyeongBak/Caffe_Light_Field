# This program writes source txt file(treating data's path) for caffe

import numpy as np
import cv2
from tqdm import tqdm

def source_txt_gen():
    TOT = 3343
    SAI = 64

    for i_sai in tqdm(range(SAI), desc='i_sai'):
        f = open('./datas/flower_dataset/train_source'+str(i_sai)+'.txt', 'w')
        for i_tot in tqdm(range(TOT), desc='i_tot'):
            #if (i_tot >= 240 and i_tot <= 250) or (i_tot >= 740 and i_tot <= 750) or (i_tot >= 2475 and i_tot <= 2485):
            #    pass
            #else:
            data = './datas/flower_dataset/flower_train_8x8/sai'+str(i_tot*3+2)+'_'+str(i_sai)+'.png 0'+'\n'
            f.write(data)
        f.close()

if __name__ == '__main__':
    source_txt_gen()