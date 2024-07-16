import json

import cv2

raw_resolution = 1080
new_resolution = 360


def generate_diff_resolution():
    path = 't1.jpg'
    img = cv2.imread(path)
    img = cv2.resize(img, (640, 360))
    cv2.imwrite('new_{}'.format(path), img)


def draw_rec():
    pic_path = 'new_t1.jpg'
    json_path = 't1_pos.json'
    img = cv2.imread(pic_path)
    with open(json_path, 'r') as f:
        boxes = json.load(f)['faces']
    modified_param = raw_resolution/new_resolution
    for box in boxes:
        cv2.rectangle(img, (int(box[0]/modified_param), int(box[1]/modified_param)), (int(box[2]/modified_param), int(box[3]/modified_param)), (0, 0, 255), 2)
    cv2.imwrite('new_res.jpg', img)


if __name__ == '__main__':
    # generate_diff_resolution()
    draw_rec()
