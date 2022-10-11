import argparse
import json
import os
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import yaml
from tqdm import tqdm

import sys
sys.path.insert(0, "../")
import cv2

from utils.datasets import letterbox
import shutil

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix
# from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized

import matplotlib.pyplot as plt
import copy
from local_utils import output_to_raw_target, draw_box_kpts, save_labelme_json


def pose_infer_imgs(opt):
    batch_size = 1
    imgsz = 960
    device = torch.device('cuda:0')
    weights = "./yolov7-w6-pose.pt"

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(imgsz, s=gs)  # check img_size

    model.eval()
    model.model[-1].flip_test = False
    model.model[-1].flip_index = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

    # Half
    half_precision = True
    half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    if half:
        model.half()

    #  proc img data
    # src_root = "/userdata/liyj/data/test_data/pose/2022-09-09-15-41-06/image_rect_color"
    # dst_root = "/userdata/liyj/data/test_data/depth/debug"

    src_root = "/userdata/liyj/data/test_data/depth/test_outdoor_person_1010/rectify/left"
    dst_root = "/userdata/liyj/data/test_data/depth/debug"

    if dst_root is not None:
        if os.path.exists(dst_root):
            shutil.rmtree(dst_root)
        os.mkdir(dst_root)

    img_names = [name for name in os.listdir(src_root) if name.split('.')[-1] in ['png', 'jpg']]
    for img_name in tqdm(img_names):
        # img_name = "image_23_1661767190924.png"
        print(img_name)
        img_path = os.path.join(src_root, img_name)
        dst_img_path = os.path.join(dst_root, img_name.replace(".png", ".jpg"))


        # img_path = "/userdata/liyj/data/test_data/pose/2022-09-09-15-41-06/image_rect_color/image_37_1662709272543.png"
        img0 = cv2.imread(img_path)  # BGR
        # img0 = img0[:, :1920, :]

        stride = int(model.stride.max())  # model stride
        # img = letterbox(img0, imgsz, stride=stride, auto=False)[0]
        img, ratio, (dw, dh) = letterbox(img0, imgsz, stride=stride, auto=False)

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():
            out, train_out = model(img, augment=False)  # inference and training outputs
            conf_thres = 0.001
            iou_thres = 0.6  # for NMS
            lb = []
            single_cls = False
            kpt_label = True
            out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls,
                                      kpt_label=kpt_label, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'])

            # show targets result
            assert len(out) == 1, "len(out) == 1"
            targets = output_to_raw_target(out[0], ratio, dw, dh)
            img_show = copy.deepcopy(img0)
            img_show = draw_box_kpts(img_show, targets)

            # save label files
            save_labelme_json(dst_img_path, img0, targets)


        # plt.imshow(img_show)
        # plt.show()
        # exit(1)


if __name__ == "__main__":
    print("Start Proc...")
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-txt-tidl', action='store_true', help='save results to *.txt in tidl format')
    parser.add_argument('--tidl-load', action='store_true', help='load thedata from a list specified as in tidl')
    parser.add_argument('--dump-img', action='store_true', help='load thedata from a list specified as in tidl')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--save-json-kpt', action='store_true', help='save a cocoapi-compatible JSON results file for key-points')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--kpt-label', action='store_true', help='Whether kpt-label is enabled or not')
    parser.add_argument('--flip-test', action='store_true', help='Whether to run flip_test or not')
    opt = parser.parse_args()
    # opt.save_json |= opt.data.endswith('coco.yaml')
    # opt.save_json_kpt |= opt.data.endswith('coco_kpts.yaml')
    # opt.data = check_file(opt.data)  # check file
    print(opt)


    pose_infer_imgs(opt)
    print("End Proc...")
