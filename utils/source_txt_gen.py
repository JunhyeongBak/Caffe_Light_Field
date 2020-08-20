# This program writes source txt file(treating data's path) for caffe

import numpy as np
import cv2
from tqdm import tqdm

def source_txt_gen():
    TOT = 800
    SAI = 25

    for i_sai in tqdm(range(SAI), desc='i_sai'):
        f = open('./datas/face_dataset/source'+str(i_sai)+'.txt', 'w')
        for i_tot in tqdm(range(TOT), desc='i_tot'):
            data = './datas/face_dataset/face_sai/sai'+str(i_tot)+'_'+str(i_sai)+'.png 0'+'\n'
            f.write(data)
        f.close()

if __name__ == '__main__':
    source_txt_gen()