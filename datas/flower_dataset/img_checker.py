import numpy as np
import cv2

for i_tot in range(3):
    for i_sai in range(64):
        src_img = cv2.imread('./datas/flower_dataset/SAIs_Crop2/sai'+str(i_tot)+'_'+str(i_sai)+'.png')
        cv2.imwrite('./datas/flower_dataset/test/sai'+str(i_tot)+'_'+str(i_sai)+'.png', src_img)