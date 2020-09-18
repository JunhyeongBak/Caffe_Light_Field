import numpy as np
import cv2

def bluring():
    img = cv2.imread('/docker/lf_depth/deploy/backup/img1/result_lf_depth_color.jpg', cv2.IMREAD_COLOR)

    kernel = np.ones((3,3), np.float32)/25
    blur = cv2.medianBlur(img, 3)
    #blur = cv2.filter2D(img, -1, kernel)

    cv2.imwrite('/docker/lf_depth/deploy/backup/img1/result_lf_depth_filt.png', blur)

bluring()