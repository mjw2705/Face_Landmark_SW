import sys
import os
import cv2
import numpy as np
import dlib

from PyQt5 import uic
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import torch

from util import *



main_ui = uic.loadUiType('sw_window.ui')[0]
fourcc = cv2.VideoWriter_fourcc(*'DIVX')

pth_path = 'ssd-mobilev1-face-2134_0.0192.pth'
dat_path = 'shape_predictor_68_face_landmarks.dat'


class MyApp(QMainWindow, main_ui):
    def __init__(self):
        super(MyApp, self).__init__()
        self.setupUi(self)
        self.initUI()

        # 변수 초기화
        self.init_dir = '../../'
        self.video_path = []

        self.face_detector = face_detector_loader(pth_path)
        self.land_detector = dlib.shape_predictor(dat_path)

        # pyqt 초기화
        self.cam_num =0
        self.cap = None
        self.press_esc = False
        self.video_frame = False
        self.get_video = False
        self.get_cam = False
        self.change_cam = False
        self.face_check = None
        self.land_check = None
        self.threshold = 0
        self.land_thres = 0

        # 버튼에 기능 연결
        self.cam_comboBox.currentIndexChanged.connect(self.camSetting_combo) # 캠 번호
        self.cam_pushButton.clicked.connect(self.camSetting_button) # 캠 선택 버튼
        self.cam_message = QMessageBox() # 캠 메시지
        self.video_pushButton.clicked.connect(self.getVideo_button) # 비디오 선택 버튼
        self.video_listWidget.itemDoubleClicked.connect(self.selectVideo) # 비디오 리스트 중 선택
        self.videoplay_pushButton.clicked.connect(self.Video_button) # 비디오 재생/정지 버튼
        # self.cali_pushButton.clicked.connect() # calibration 버튼

        # 체크박스
        self.face_checkBox.stateChanged.connect(self.check_face)
        self.land_checkBox.stateChanged.connect(self.check_land)
        self.pupil_checkBox.stateChanged.connect(self.check_pupil)

        # slider
        self.thres_horizontalSlider.setValue(self.threshold)
        self.landthres_horizontalSlider.setValue(self.land_thres)
        self.thres_label.setText(str(self.threshold))
        self.land_label.setText(str(self.land_thres))
        self.thres_horizontalSlider.setRange(0, 255)
        self.landthres_horizontalSlider.setRange(0, 20)
        self.thres_horizontalSlider.valueChanged.connect(self.thres_slider)
        self.landthres_horizontalSlider.valueChanged.connect(self.landthres_slider)

        self.exit_Button.clicked.connect(self.prgram_exit) # 종료 버튼

    def thres_slider(self):
        self.threshold = self.thres_horizontalSlider.value()
        self.thres_label.setText(str(self.threshold))

    def landthres_slider(self):
        self.land_thres = self.landthres_horizontalSlider.value()
        self.land_label.setText(str(self.land_thres))

    def check_face(self):
        if self.face_checkBox.isChecked():
            return True
        else:
            return False

    def check_land(self):
        if self.land_checkBox.isChecked():
            return True
        else:
            return False

    def check_pupil(self):
        if self.pupil_checkBox.isChecked():
            return True
        else:
            return False

    def initUI(self):
        self.setWindowTitle('1-stage detection')
        # self.setWindowIcon()
        self.cam_comboBox.addItem('0')
        self.cam_comboBox.addItem('1')
        self.cam_comboBox.addItem('2')
        self.cam_comboBox.addItem('3')
        self.center()
        self.show()

    def camSetting_combo(self):
        self.change_cam = True
        self.cam_num = int(self.cam_comboBox.currentText())

    def camSetting_button(self):
        self.get_cam = True
        self.change_cam = False
        self.get_video = False

        self.cap = cv2.VideoCapture(self.cam_num, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            self.cam_message.setWindowTitle('Message')
            self.cam_message.setIcon(QMessageBox.Information)
            self.cam_message.setText('This cam is not work')
            self.cam_message.setStandardButtons(QMessageBox.Ok)
            retval = self.cam_message.exec_()

            self.cap = None
        else:
            self.startCamera()
            print('camera stop')

    def startCamera(self):
        self.get_cam = True

        if self.cap:
            face_detect, land_detect = False, False
            prev_bbox, prev_land = [], []
            while True:
                self.ret, self.frame = self.cap.read()
                if self.ret:
                    self.detect(self.frame, prev_bbox, face_detect, prev_land, land_detect, 1)

                    self.showImage(self.frame, self.display_label)
                    cv2.waitKey(1)

                    # 비디오가 눌리면 / cam 바뀌면 stop
                    if self.press_esc or self.get_video or self.change_cam:
                        self.cap.release()
                        break
                else:
                    break


    def getVideo_button(self):
        self.get_cam = False
        self.get_video = True
        self.video_path = QFileDialog.getOpenFileNames(self, 'Select video', self.init_dir)[0]
        print(self.video_path)

        if self.video_path:
            for i, path in enumerate(self.video_path):
                self.video_listWidget.insertItem(i, os.path.basename(path))
        else:
            self.video_listWidget.clear()

    def selectVideo(self):
        self.get_cam = False
        self.get_video = True
        self.idx = self.video_listWidget.currentRow()
        self.cap = cv2.VideoCapture(self.video_path[self.idx])

        self.ret, self.frame = self.cap.read()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if self.ret:
            self.video_frame = True
            self.showImage(self.frame, self.display_label)

    def Video_button(self):
        if self.video_frame:
            self.video_frame = False
            self.startVideo()
            self.get_cam = False
        else:
            self.video_frame = True

    def startVideo(self):
        if self.cap:
            face_detect, land_detect = False, False
            prev_bbox, prev_land = [], []

            while True:
                self.ret, self.frame = self.cap.read()

                if self.ret and not self.video_frame:
                    self.detect(self.frame, prev_bbox, face_detect, prev_land, land_detect, 2)

                    self.showImage(self.frame, self.display_label)
                    cv2.waitKey(1)

                # cam이 눌리면 stop
                elif self.press_esc or self.get_cam:
                    self.cap.release()
                    break

                else:
                    break


    def detect(self, frame, prev_bbox, face_detect, prev_land, land_detect, thick):
        boxes, labels, probs, sec1 = get_face(self.face_detector, frame)

        if boxes.size(0):
            '''detect face'''
            box = boxes[0, :]
            label = f"Face: {probs[0]:.2f}"
            cur_bbox = add_face_region(box)
            cur_bbox, prev_bbox, face_detect = low_pass_filter(cur_bbox, prev_bbox, face_detect,
                                                               mode='face')
            x1, x2, y1, y2 = cur_bbox

            '''detect landmark'''
            face_box = dlib.rectangle(left=cur_bbox[0], top=cur_bbox[2], right=cur_bbox[1],
                                      bottom=cur_bbox[3])
            cur_land = self.land_detector(self.frame, face_box)
            cur_land = cvt_shape_to_np(cur_land, land_add=self.land_thres)
            cur_land, prev_land, land_detect = low_pass_filter(cur_land, prev_land, land_detect, mode='landmark')
            cur_rel_coord = cvt_land_rel(cur_land, cur_bbox)

            '''eyeball'''
            face = self.frame[y1:y2, x1:x2].copy()
            face = cv2.resize(face, (300, 300))

            abs_land = (cur_rel_coord * face_size).astype(np.int)

            mask = np.zeros(face.shape[:2], dtype=np.uint8)
            mask = eye_on_mask(abs_land, mask, left_eye)
            mask = eye_on_mask(abs_land, mask, right_eye)
            kernel = np.ones((9, 9), np.uint8)
            mask = cv2.dilate(mask, kernel, 5)
            eyes = cv2.bitwise_and(face, face, mask=mask)
            mask = (eyes == [0, 0, 0]).all(axis=2)
            eyes[mask] = [255, 255, 255]
            eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)

            _, thresh = cv2.threshold(eyes_gray, self.threshold, 255, cv2.THRESH_BINARY)
            thresh = cv2.erode(thresh, None, iterations=2)
            thresh = cv2.dilate(thresh, None, iterations=4)
            thresh = cv2.medianBlur(thresh, 3)
            thresh = cv2.bitwise_not(thresh)
            mid = (abs_land[42][0] + abs_land[39][0]) // 2
            l_center = contouring(thresh[:, 0:mid], mid, face)
            r_center = contouring(thresh[:, mid:], mid, face, True)

            cv2.imshow('image', thresh)

            if self.check_pupil() == True:
            #     if 0 not in centers[0] and 0 not in centers[1]:
                l_center = np.array(l_center) / face.shape[0]
                r_center = np.array(r_center) / face.shape[0]
                frame = cv2.circle(frame, (int(l_center[0] * (x2 - x1) + x1), int(l_center[1] * (y2 - y1) + y1)), 2,
                                   (255, 255, 255), -1)
                frame = cv2.circle(frame, (int(r_center[0] * (x2 - x1) + x1), int(r_center[1] * (y2 - y1) + y1)), 2,
                                   (255, 255, 255), -1)

            if self.check_face() == True:
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 0, 255), 4)
                frame = cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                         (255, 255, 255), 2)

            if self.check_land() == True:
                frame = draw_land(frame, cur_land, (0, 0, 255), thick)
                # for i in range(68):
                #     x, y = cur_land.part(i).x, cur_land.part(i).y
                #     cv2.circle(frame, (x, y), thick, (0, 0, 255), -1)


    def showImage(self, img, display_label):
        draw_img = img.copy()
        height = display_label.height()
        width = display_label.width()
        bytesPerLine = 3 * width

        draw_img = cv2.resize(draw_img, (height, width))

        qt_image = QImage(draw_img.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        qpixmap = QPixmap.fromImage(qt_image)
        display_label.setPixmap(qpixmap)

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def prgram_exit(self):
        self.press_esc = True
        QCoreApplication.instance().quit()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    my_window = MyApp()
    my_window.show()
    sys.exit(app.exec_())