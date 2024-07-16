import json

import cv2
import sys
import numpy as np
import datetime
import os
import glob
from retinaface import RetinaFace


def generate_face_detection_annotation(path, out, validate):
    thresh = 0.8
    gpuid = 0
    detector = RetinaFace('./model/R50', 0, gpuid, 'net3')

    res = []
    list_dirs = sorted(os.listdir(path), key=lambda x: os.path.getmtime(os.path.join(path, x)))
    for pic in list_dirs:
        if pic.endswith('.jpg'):
            pic_res = []
            img = cv2.imread(os.path.join(path, pic))
            faces, landmarks = detector.detect(img,
                                               thresh,
                                               scales=[1],
                                               do_flip=False)
            if faces is not None:
                print(f'find {faces.shape[0]} faces in {os.path.join(path, pic)}')
                for i in range(faces.shape[0]):
                    # print('score', faces[i][4])
                    box = faces[i].astype(np.int)
                    color = (0, 0, 255)
                    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
                    pic_res.append({'detection': [int(box[0]), int(box[1]), int(box[2]), int(box[3])]})

                    if landmarks is not None:
                        landmark5 = landmarks[i].astype(np.int)
                        # print(landmark.shape)
                        for l in range(landmark5.shape[0]):
                            color = (0, 0, 255)
                            if l == 0 or l == 3:
                                color = (0, 255, 0)
                            cv2.circle(img, (landmark5[l][0], landmark5[l][1]), 1, color, 2)

            cv2.imwrite(os.path.join(validate, pic), img)
            res.append({'name': pic, 'annotation': pic_res})

    with open(out, 'w') as f:
        json.dump(res, f)


def generate_validate_video(input_path, output_path):
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (1920, 1080))
    list_dirs = sorted(os.listdir(input_path), key=lambda x: os.path.getmtime(os.path.join(input_path, x)))
    for pic in list_dirs:
        if pic.endswith('.jpg'):
            out.write(cv2.imread(os.path.join(input_path, pic)))
    out.release()


if __name__ == '__main__':
    generate_face_detection_annotation(r'G:\PycharmProjects\Auto-Schedule\Code\dataset\train',
                                       r'G:\PycharmProjects\Auto-Schedule\Code\dataset\annotation_train.json',
                                       r'G:\PycharmProjects\Auto-Schedule\Code\dataset\validate\train')
    generate_face_detection_annotation(r'G:\PycharmProjects\Auto-Schedule\Code\dataset\test',
                                       r'G:\PycharmProjects\Auto-Schedule\Code\dataset\annotation_test.json',
                                       r'G:\PycharmProjects\Auto-Schedule\Code\dataset\validate\test')

    # generate_validate_video(r'G:\PycharmProjects\Auto-Schedule\Code\dataset\validate\train',
    #                         r'G:\PycharmProjects\Auto-Schedule\Code\dataset\validate\train.mp4')
    # generate_validate_video(r'G:\PycharmProjects\Auto-Schedule\Code\dataset\validate\test',
    #                         r'G:\PycharmProjects\Auto-Schedule\Code\dataset\validate\test.mp4')
