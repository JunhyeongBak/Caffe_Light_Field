# -*- coding: utf-8 -*-
from PyQt4 import QtCore, QtGui, uic
import sys
import cv2
import numpy as np
import threading
import time
import Queue
#import denseUNet
np.set_printoptions(threshold=sys.maxsize)
running = False
capture_thread = None
form_class = uic.loadUiType("simple.ui")[0]
q = Queue.Queue()

SHIFT_VAL = -1.4 
DEPLOY_PATH = './scripts/denseUNet_deploy.prototxt'
MODEL_PATH = './models'
#denseUNet.denseUNet_deploy(DEPLOY_PATH, tot=1207, sai=25, batch_size=1, shift_val=SHIFT_VAL)


face_mask = cv2.imread('./face_color.png', 1)
face_mask = cv2.resize(face_mask, (468, 351), interpolation = cv2.INTER_CUBIC)
face_mask_gray = cv2.imread('./face_color.png', 0)
face_mask_gray = cv2.resize(face_mask_gray, (468, 351), interpolation = cv2.INTER_CUBIC)

ret, mask = cv2.threshold(face_mask_gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

mask = mask//255
mask_inv = mask_inv//255

def grab(cam, queue, width, height, fps):
    global running
    capture = cv2.VideoCapture(cam)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)

    while(running):
        frame = {}        
        capture.grab()
        retval, img = capture.retrieve(0)
        frame["img"] = img

        if queue.qsize() < 10:
            queue.put(frame)
        #else:
            #print queue.qsize()

class OwnImageWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        super(OwnImageWidget, self).__init__(parent)
        self.image = None

    def setImage(self, image):
        self.image = image
        sz = image.size()
        self.setMinimumSize(sz)
        self.update()

    def paintEvent(self, event):
        qp = QtGui.QPainter()
        qp.begin(self)
        if self.image:
            qp.drawImage(QtCore.QPoint(0, 0), self.image)
        qp.end()

class MyWindowClass(QtGui.QMainWindow, form_class):
    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self, parent)
        self.setupUi(self)

       # self.startButton.clicked.connect(self.start_clicked)
        global running
        running = True
        capture_thread.start()

        self.window_width = self.ImgWidget.frameSize().width()
        self.window_height = self.ImgWidget.frameSize().height()
        self.ImgWidget = OwnImageWidget(self.ImgWidget)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1)

        self.startButton.clicked.connect(self.botton_clicked)

        self.mode_flag = 'Render'
        self.render_init = False
        self.render_cnt = 0
        self.img_list = None
        self.test_cnt = 0

    def botton_clicked(self):
        if self.mode_flag == 'Render':
            self.mode_flag = 'Camera'
        else:
            self.mode_flag = 'Render'
            self.render_init = False
        self.startButton.setText(self.mode_flag)

    def update_frame(self):
        if self.mode_flag == 'Render':
            if not q.empty():
                frame = q.get()
                img = frame["img"]

                img_height, img_width, img_colors = img.shape
                scale_w = float(self.window_width) / float(img_width)
                scale_h = float(self.window_height) / float(img_height)
                scale = min([scale_w, scale_h])

                if scale == 0:
                    scale = 1
                
                img = cv2.resize(img, None, fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)

                print(img.shape)
                print(mask_inv.shape)


                img1_bg = cv2.bitwise_and(img, img, mask=mask_inv)
                img2_fg = cv2.bitwise_and(face_mask, face_mask, mask=mask)
                img = cv2.add(img1_bg, img2_fg)



                img_masked = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                height, width, bpc = img_masked.shape
                bpl = bpc * width
                image = QtGui.QImage(img_masked.data, width, height, bpl, QtGui.QImage.Format_RGB888)
                self.ImgWidget.setImage(image)
        else:
            if self.render_init == False:
                frame = q.get()
                img_frame = frame["img"]
                img_frame = img_frame[:, 80:80+480, :]
                img_frame = cv2.resize(img_frame, (192, 192), interpolation=cv2.INTER_CUBIC)
                #test_img = cv2.imread('./test.png', 1)
                self.img_list = denseUNet.denseUNet_runner(DEPLOY_PATH, MODEL_PATH+'/denseUNet_solver_iter_44659.caffemodel', img_frame)
                self.test_cnt = self.test_cnt + 1
                self.render_init = True

            img = self.img_list[:, :, :, self.render_cnt]
            img = np.array(img, dtype=np.uint8)
            cv2.imwrite('./datas/face_dataset/result/sai'+str(self.test_cnt)+'_'+str(self.render_cnt)+'.png', img)
            self.render_cnt = self.render_cnt + 1
            if self.render_cnt == 25:
                self.render_cnt = 0

            

            img_height, img_width, img_colors = img.shape
            scale_w = float(self.window_width) / float(img_width)
            scale_h = float(self.window_height) / float(img_height)
            scale = min([scale_w, scale_h])

            if scale == 0:
                scale = 1
            
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, bpc = img.shape
            bpl = bpc * width
            image = QtGui.QImage(img.data, width, height, bpl, QtGui.QImage.Format_RGB888)
            self.ImgWidget.setImage(image)

            time.sleep(0.05)




    def closeEvent(self, event):
        global running
        running = False



capture_thread = threading.Thread(target=grab, args = (0, q, 1920, 1080, 30))

app = QtGui.QApplication(sys.argv)
w = MyWindowClass(None)
w.setWindowTitle('DenseUNet Demo')
w.show()
app.exec_()
