import sys
import os
import cv2
import numpy as np
import dlib

from PyQt5 import uic
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from sklearn.linear_model import LinearRegression
from collections import defaultdict

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
        self.click_pt = []
        self.reg = LinearRegression()
        self.centors = None
        self.vectors = defaultdict(list)

        self.face_detector = face_detector_loader(pth_path)
        self.land_detector = dlib.shape_predictor(dat_path)

        # pyqt 초기화
        self.cam_num = 0
        self.cap = None
        self.press_esc = False
        self.video_frame = False
        self.get_video = False
        self.get_cam = False
        self.change_cam = False
        self.face_check = None
        self.land_check = None
        self.calibration = False
        self.cali_complete = False
        self.threshold = 50
        self.land_thres = 4

        # 버튼에 기능 연결
        self.cam_comboBox.currentIndexChanged.connect(self.camSetting_combo)  # 캠 번호
        self.cam_pushButton.clicked.connect(self.camSetting_button)  # 캠 선택 버튼
        self.cam_message = QMessageBox()  # 캠 메시지
        self.video_pushButton.clicked.connect(self.getVideo_button)  # 비디오 선택 버튼
        self.video_listWidget.itemDoubleClicked.connect(self.selectVideo)  # 비디오 리스트 중 선택
        self.videoplay_pushButton.clicked.connect(self.Video_button)  # 비디오 재생/정지 버튼
        self.cali_pushButton.clicked.connect(self.calibration_button)  # calibration 버튼

        # 체크박스
        self.gaze_checkBox.setEnabled(False)
        self.face_checkBox.stateChanged.connect(self.check_face)
        self.land_checkBox.stateChanged.connect(self.check_land)
        self.pupil_checkBox.stateChanged.connect(self.check_pupil)
        self.gaze_checkBox.stateChanged.connect(self.check_gaze)

        # slider
        self.thres_horizontalSlider.setValue(self.threshold)
        self.landthres_horizontalSlider.setValue(self.land_thres)
        self.thres_label.setText(str(self.threshold))
        self.land_label.setText(str(self.land_thres))
        self.thres_horizontalSlider.setRange(0, 255)
        self.landthres_horizontalSlider.setRange(0, 20)
        self.thres_horizontalSlider.valueChanged.connect(self.thres_slider)
        self.landthres_horizontalSlider.valueChanged.connect(self.landthres_slider)

        self.exit_Button.clicked.connect(self.program_exit)  # 종료 버튼

    def initUI(self):
        self.setWindowTitle('detection')
        # self.setWindowIcon()
        self.cam_comboBox.addItem('0')
        self.cam_comboBox.addItem('1')
        self.cam_comboBox.addItem('2')
        self.cam_comboBox.addItem('3')
        self.center()
        self.show()

    def thres_slider(self):
        self.threshold = self.thres_horizontalSlider.value()
        self.thres_label.setText(str(self.threshold))

    def landthres_slider(self):
        self.land_thres = self.landthres_horizontalSlider.value()
        self.land_label.setText(str(self.land_thres))

    def click_event(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_pt = [x, y]

    def calibration_button(self):
        self.calibration = True

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

    def check_gaze(self):
        if self.gaze_checkBox.isChecked():
            return True
        else:
            return False

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
            self.cap = None
        else:
            self.startCamera()
            print('camera stop')

    def compute(self, vector):
        np_vector = np.array([vector])
        np_gaze = self.reg.predict(np_vector)

        return np_gaze

    def get_gaze(self, vector):
        gaze = None
        if vector:
            gaze = self.compute(vector)

        return gaze

    def gaze_tracking(self, cur_center, type='full'):
        width = self.display_label.width()
        height = self.display_label.height()

        gaze = self.get_gaze(cur_center)

        gaze_x, gaze_y = gaze[0][0], gaze[0][1]

        gaze_x = 1 if gaze_x > 1 else gaze_x
        gaze_y = 1 if gaze_y > 1 else gaze_y
        gaze_x = 0 if gaze_x < 0 else gaze_x
        gaze_y = 0 if gaze_y < 0 else gaze_y

        # 전체 이미지
        if type == 'full':
            gaze_x = int(gaze_x * width - 10)
            gaze_y = int(gaze_y * height - 10)

            print('center: ', cur_center, 'predict: ', (gaze_x, gaze_y))
            gaze_x = 10 if gaze_x < 10 else gaze_x
            gaze_x = width - 30 if gaze_x > width - 30 else gaze_x
            gaze_y = 10 if gaze_y < 10 else gaze_y
            gaze_y = height - 30 if gaze_y > height - 30 else gaze_y

            # 전체 이미지에서 뿌리기기
            frame = cv2.resize(self.frame, (height, width))
            frame = cv2.flip(frame, 1)
            frame = cv2.circle(frame, (gaze_x, gaze_y), 10, (0, 0, 255), -1)

            cv2.namedWindow("gaze_whiteboard", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("gaze_whiteboard", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("gaze_whiteboard", frame)

            if cv2.waitKey(1) == 27:
                self.gaze_checkBox.toggle()
                cv2.destroyWindow("gaze_whiteboard")

        else:
            # pyqt
            gaze_x = int(gaze_x * width)
            gaze_y = int(gaze_y * height)
            gaze_x = 10 if gaze_x < 10 else gaze_x
            gaze_x = width - 30 if gaze_x > width - 30 else gaze_x
            gaze_y = 10 if gaze_y < 10 else gaze_y
            gaze_y = height - 30 if gaze_y > height - 30 else gaze_y

            # pyqt에서 뿌리기
            # self.frame = cv2.resize(self.frame, (height, width))
            self.frame = cv2.circle(self.frame, (gaze_x, gaze_y), 10, (0, 0, 255), -1)


    def calibration_board(self, i, j):
        width = self.display_label.width()
        height = self.display_label.height()

        whiteboard = np.ones((height, width, 3))

        w_interval = (width // (points[0] ** 0.5 - 1))
        h_interval = (height // (points[0] ** 0.5 - 1))
        x = int(init_x + (w_interval * i) - 10)
        y = int(init_y + (h_interval * j) - 10)
        x = 10 if x < 10 else x
        x = width - 10 if x > width - 10 else x
        y = 10 if y < 10 else y
        y = height - 10 if y > height - 10 else y

        whiteboard = cv2.circle(whiteboard, (x, y), 10, (0, 0, 255), -1)

        cv2.namedWindow("whiteboard", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("whiteboard", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("whiteboard", whiteboard)
        cv2.setMouseCallback("whiteboard", self.click_event)

        if cv2.waitKey(1) == 27:
            cv2.destroyWindow("whiteboard")
            self.calibration = False

        return x, y

    def regression_model(self):
        all_v = []
        all_p = []

        for eye_cor, point in self.vectors.items():
            eye = np.array(eye_cor)
            pt = np.array(point[0])

            all_v.append(eye)
            all_p.append(pt)

        self.reg.fit(all_v, all_p)
        self.cali_complete = False

    def startCamera(self):
        self.get_cam = True

        if self.cap:
            prev_bbox, prev_land, prev_center = [0, 0, 0, 0], np.zeros((68, 2)), [0, 0, 0, 0]
            width = self.display_label.width()
            height = self.display_label.height()
            n_point, i, j, enough = 0, 0, 0, 0

            while True:
                self.ret, self.frame = self.cap.read()
                if self.ret:
                    boxes, labels, probs, sec1 = get_face(self.face_detector, self.frame)

                    if boxes.size(0):
                        label = f"Face: {probs[0]:.2f}"
                        face_detect, land_detect, eye_detect = True, True, True
                        cur_bbox = self.detect_face(boxes, prev_bbox, face_detect)
                        cur_land = self.detect_lands(self.frame, cur_bbox, prev_land, land_detect)
                        cur_center = self.detect_eyeball(self.frame, cur_bbox, cur_land, prev_center, eye_detect)

                        self.draw(self.frame, label, cur_bbox, cur_land, cur_center, 1)

                        cur_center = np.array(cur_center)
                        cur_center = [cur_center[0] / width, cur_center[1] / height,
                                      cur_center[2] / width, cur_center[3] / height]

                        # 캘리브레이션
                        if self.calibration == True:
                            self.gaze_checkBox.setEnabled(True)
                            if j >= int(points[n_point] ** 0.5):
                                j = 0
                            if i >= int(points[n_point] ** 0.5):
                                i = 0
                                j += 1

                            x, y = self.calibration_board(i, j)

                            # 클릭 포인트 저장
                            if self.click_pt:
                                if abs(self.click_pt[0] - x) < 10 and abs(self.click_pt[1] - y) < 10:
                                    cal_pt = f'cal_{points[0]}_{j}_{i}'
                                    print(cal_pt)
                                    print(f'center: {cur_center}')
                                    print(f'x : {self.click_pt[0]}, y : {self.click_pt[1]}')
                                    i += 1

                                    self.click_pt = (self.click_pt[0] / width, self.click_pt[1] / height)
                                    self.vectors[tuple(cur_center)].append(self.click_pt)
                                    enough += 1

                                    self.click_pt = []
                                else:
                                    self.click_pt = []

                            if enough >= points[0]:
                                cv2.destroyWindow("whiteboard")
                                self.calibration = False
                                self.cali_complete = True

                        # 선형 회귀 모델로 학습
                        if self.cali_complete == True:
                            self.regression_model()

                        # gaze tracking 창 띄우기
                        if self.check_gaze() == True and len(self.vectors) == points[0]:
                            self.gaze_tracking(cur_center, 'pyqt')


                    self.showImage(self.frame, self.display_label)
                    cv2.waitKey(1)

                    # 비디오가 눌리면 / cam 바뀌면 stop
                    if self.press_esc or self.get_video or self.change_cam:
                        # cv2.destroyWindow("gaze_whiteboard")
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
            prev_bbox, prev_land, prev_center = [0, 0, 0, 0], np.zeros((68, 2)), [0, 0, 0, 0]

            while True:
                self.ret, self.frame = self.cap.read()

                if self.ret and not self.video_frame:
                    boxes, labels, probs, sec1 = get_face(self.face_detector, self.frame)

                    if boxes.size(0):
                        label = f"Face: {probs[0]:.2f}"
                        face_detect, land_detect, eye_detect = True, True, True
                        cur_bbox = self.detect_face(boxes, prev_bbox, face_detect)
                        cur_land = self.detect_lands(self.frame, cur_bbox, prev_land, land_detect)
                        cur_center = self.detect_eyeball(self.frame, cur_bbox, cur_land, prev_center, eye_detect)

                        self.draw(self.frame, label, cur_bbox, cur_land, cur_center, 2)

                    self.showImage(self.frame, self.display_label)
                    cv2.waitKey(1)

                # cam이 눌리면 stop
                elif self.press_esc or self.get_cam:
                    self.cap.release()
                    break

                else:
                    break

    def detect_face(self, boxes, prev_bbox, face_detect):
        box = boxes[0, :]

        cur_bbox = add_face_region(box)
        cur_bbox, prev_bbox, face_detect = low_pass_filter(cur_bbox, prev_bbox, face_detect, mode='face')

        return cur_bbox

    def detect_lands(self, frame, cur_bbox, prev_land, land_detect):
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_box = dlib.rectangle(left=cur_bbox[0], top=cur_bbox[2], right=cur_bbox[1], bottom=cur_bbox[3])
        cur_land = self.land_detector(gray_img, face_box)
        cur_land = cvt_shape_to_np(cur_land, land_add=self.land_thres)
        cur_land, prev_land, land_detect = low_pass_filter(cur_land, prev_land, land_detect, mode='landmark')

        return cur_land

    def detect_eyeball(self, frame, cur_bbox, cur_land, prev_center, eye_detect):
        x1, x2, y1, y2 = cur_bbox
        cur_rel_coord = cvt_land_rel(cur_land, cur_bbox)

        face = frame[y1:y2, x1:x2].copy()
        face = cv2.resize(face, (face_size, face_size))

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
        cv2.imshow('image', thresh)
        mid = (abs_land[42][0] + abs_land[39][0]) // 2
        l_center = contouring(thresh[:, 0:mid], mid, face)
        r_center = contouring(thresh[:, mid:], mid, face, True)

        l_center = np.array(l_center) / face.shape[0]
        r_center = np.array(r_center) / face.shape[0]

        pt_l_center = [int(l_center[0] * (x2 - x1) + x1), int(l_center[1] * (y2 - y1) + y1)]
        pt_r_center = [int(r_center[0] * (x2 - x1) + x1), int(r_center[1] * (y2 - y1) + y1)]

        cur_center, prev_cen, land_detect = low_pass_filter_eyecenter(pt_l_center + pt_r_center, prev_center, eye_detect)

        return cur_center

    def draw(self, frame, label, cur_bbox, cur_land, cur_center, thick=2):
        x1, x2, y1, y2 = cur_bbox
        if self.check_pupil() == True:
            frame = cv2.circle(frame, (cur_center[0], cur_center[1]), 2, (255, 255, 255), -1)
            frame = cv2.circle(frame, (cur_center[2], cur_center[3]), 2, (255, 255, 255), -1)

        if self.check_face() == True:
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 0, 255), 4)
            frame = cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 2)

        if self.check_land() == True:
            for (x, y) in cur_land:
                cv2.circle(frame, (x, y), thick, (0, 0, 255), -1)


    def showImage(self, img, display_label):
        qpixmap = cvtPixmap(img, (display_label.width(), display_label.height()))
        display_label.setPixmap(qpixmap)

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def program_exit(self):
        self.press_esc = True
        QCoreApplication.instance().quit()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    my_window = MyApp()
    my_window.show()
    sys.exit(app.exec_())
