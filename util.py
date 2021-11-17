import cv2
import time
import dlib
import numpy as np

from model.mobilenetv1 import MobileNetV1
from model.ssd import SSD, Predictor

import torch

from PyQt5.QtGui import *

# setting
face_size = 300
bbox_region = {'forehead': 35, 'chin': 0, 'add_face_width': 10}
filters = {'bbox': 20, 'landmark': 3, 'center': 2}

left_eye = [36, 37, 38, 39, 40, 41]
right_eye = [42, 43, 44, 45, 46, 47]
eye_top = [37, 38,  43, 44]
eye_bottom = [40, 41, 46, 47]
eye_side_left = [36, 42]
eye_side_right = [39, 45]

points = [9]
# points = [9, 16, 25]
init_x, init_y = 10, 10


def nothing(x):
    pass


def face_detector_loader(pth_path):
    f_detection_model = SSD(2, MobileNetV1(2), is_training=False)
    state = torch.load(pth_path)
    f_detection_model.load_state_dict(state['model_state_dict'])
    predictor = Predictor(f_detection_model, 300)

    return predictor


def get_face(detector, frame):
    # frame = cv2.resize(frame, (640, 480))
    prevTime = time.time()
    boxes, labels, probs = detector.predict(frame, 1, 0.5)
    sec = time.time() - prevTime

    return boxes, labels, probs, sec


def add_face_region(box):
    x1, x2, y1, y2 = int(box[0].item() - bbox_region['add_face_width']), int(
        box[2].item() + bbox_region['add_face_width']), int(box[1].item() + bbox_region['forehead']), int(
        box[3].item() + bbox_region['chin'])
    x1 = 0 if x1 < 0 else x1

    return [x1, x2, y1, y2]


def low_pass_filter(cur, prev, detect, mode=None):
    if mode == 'face':
        if detect:
            if abs(prev[0] - cur[0]) < filters['bbox']:
                cur[0] = prev[0]
            else:
                prev[0] = cur[0]
            if abs(prev[1] - cur[1]) < filters['bbox']:
                cur[1] = prev[1]
            else:
                prev[1] = cur[1]
            if abs(prev[2] - cur[2]) < filters['bbox']:
                cur[2] = prev[2]
            else:
                prev[2] = cur[2]
            if abs(prev[3] - cur[3]) < filters['bbox']:
                cur[3] = prev[3]
            else:
                prev[3] = cur[3]
        else:
            detect = True
            prev = cur
    elif mode == 'landmark':
        if detect:
            idx = 0
            for land, prev_land in zip(cur, prev):
                if abs(land[0] - prev_land[0]) < filters['landmark']:
                    cur[idx][0] = prev_land[0]
                else:
                    prev[idx][0] = land[0]
                if abs(land[1] - prev_land[1]) < filters['landmark']:
                    cur[idx][1] = prev_land[1]
                else:
                    prev[idx][1] = land[1]
                idx += 1
        else:
            detect = True
            prev = cur

    return cur, prev, detect


def low_pass_filter_eyecenter(cur, prev, detect):
    if detect:
        if abs(prev[0] - cur[0]) < filters['center']:
            cur[0] = prev[0]
        else:
            prev[0] = cur[0]
        if abs(prev[1] - cur[1]) < filters['center']:
            cur[1] = prev[1]
        else:
            prev[1] = cur[1]
        if abs(prev[2] - cur[2]) < filters['center']:
            cur[2] = prev[2]
        else:
            prev[2] = cur[2]
        if abs(prev[3] - cur[3]) < filters['center']:
            cur[3] = prev[3]
        else:
            prev[3] = cur[3]

    else:
        detect = True
        prev = cur

    return cur, prev, detect


def cvt_shape_to_np(landmakrs, land_add=0, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        x = landmakrs.part(i).x
        y = landmakrs.part(i).y
        if i in eye_top:
            coords[i] = (x, y - land_add)
        elif i in eye_bottom:
            coords[i] = (x, y + land_add)
        elif i in eye_side_left:
            coords[i] = (x - (land_add-2), y)
        elif i in eye_side_right:
            coords[i] = (x + (land_add+2), y)
        else:
            coords[i] = (x, y)

    return coords


def cvt_land_rel(land, cur_box):
    rel_land = np.zeros((68, 2), dtype="float")

    rel_land[:, 0] = (land[:, 0] - cur_box[0]) / (cur_box[1] - cur_box[0])
    rel_land[:, 1] = (land[:, 1] - cur_box[2]) / (cur_box[3] - cur_box[2])

    return rel_land


def eye_on_mask(landmarks, mask, side):
    points = [landmarks[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    return mask

def cvtPixmap(frame, img_size):
    frame = cv2.resize(frame, img_size)
    height, width, channel = frame.shape
    bytesPerLine = 3 * width
    qImg = QImage(frame.data,
                  width,
                  height,
                  bytesPerLine,
                  QImage.Format_RGB888).rgbSwapped()
    qpixmap = QPixmap.fromImage(qImg)

    return qpixmap


def contouring(thresh, mid, img, right=False):
    cx, cy = 0, 0
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    try:
        cnt = max(cnts, key=cv2.contourArea)  # finding contour with #maximum area
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        if right:
            cx += mid  # Adding value of mid to x coordinate of centre of #right eye to adjust for dividing into two parts
        cv2.circle(img, (cx, cy), 2, (0, 0, 255), -1)  # drawing over #eyeball with red
    except:
        pass

    return cx, cy