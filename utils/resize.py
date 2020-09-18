import numpy as np
import cv2

TOT = 800
SAI = 25
HEIGHT = 192*4
WIDTH = 192*4
CH = 3

if __name__ == '__main__':
    for i_tot in range(TOT):
        for i_sai in range(SAI):
            src_img = cv2.imread('./2_blender_LF/val_output_data/sai'+str(i_tot)+'_'+str(i_sai)+'.png', cv2.IMREAD_COLOR)
            dst_img = cv2.resize(src_img, (192, 192), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite('./2_blender_LF/val_small_set/sai'+str(i_tot)+'_'+str(i_sai)+'.png', dst_img)