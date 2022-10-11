import sys
sys.path.insert(0, "../")

import numpy as np
import copy
import cv2
import os
import json


def output_to_raw_target(output, ratio, dw, dh):
    # img_show = copy.deepcopy(img_raw)
    # box
    output[:, 0:4:2] = (output[:, 0:4:2] - dw)/ratio[0]
    output[:, 1:4:2] = (output[:, 1:4:2] - dh)/ratio[1]

    output[:, 6::3] = (output[:, 6::3] - dw)/ratio[0]
    output[:, 7::3] = (output[:, 7::3] - dh)/ratio[1]

    return output


def draw_box_kpts(img, targets, kp_steps=3):
    for target in targets:
        box = target[:4]
        conf = target[4]
        cls = target[5]
        kps = target[6:]

        num_kpts = len(kps) // kp_steps

        if conf > 0.2:
            c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(img, c1, c2, (255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

            for kid in range(num_kpts):
                kp_x, kp_y = kps[kp_steps * kid], kps[kp_steps * kid + 1]
                kp_x = int(kp_x + 0)
                kp_y = int(kp_y + 0)
                cv2.circle(img, (kp_x, kp_y), int(5), (0, 255, 0), -1)
    return img


def save_labelme_json(img_path, img, targets, kp_steps=3):
    img_name = os.path.split(img_path)[-1]
    img_h, img_w, _ = img.shape
    final_coor = {"imagePath": img_name, "imageData": None, "shapes": [], "version": "3.5.0",
                  "flags": None, "fillColor": [255, 0, 0, 128], "lineColor": [0, 255, 0, 128], \
                  "imageWidth": img_w, "imageHeight": img_h}

    for i, target in enumerate(targets):
        box = target[:4]
        conf = target[4]
        cls = target[5]
        kps = target[6:]

        num_kpts = len(kps) // kp_steps
        if conf > 0.2:
            # box anns
            box_name = "_".join(["cls", str(int(cls)), "id", str(i)])
            labelme_box = {"shape_type": "rectangle", "line_color": None, "points": [], \
                            "fill_color": None, "label": box_name}
            labelme_box["points"].append([int(box[0]), int(box[1])])
            labelme_box["points"].append([int(box[2]), int(box[3])])

            final_coor["shapes"].append(labelme_box)

            # points
            for kid in range(num_kpts):
                kp_x, kp_y = kps[kp_steps * kid], kps[kp_steps * kid + 1]
                point_name = "_".join(["kp", str(int(kid)), "id", str(i)])
                labelme_point = {"shape_type": "point", "line_color": None, "points": [], \
                               "fill_color": None, "label": point_name}
                labelme_point["points"].append([int(kp_x), int(kp_y)])

                final_coor["shapes"].append(labelme_point)
    # print(final_coor["shapes"])
    save_label_me_path = os.path.splitext(img_path)[0] + ".json"
    with open(save_label_me_path, 'w') as fp:
        json.dump(final_coor, fp)

    cv2.imwrite(img_path, img)