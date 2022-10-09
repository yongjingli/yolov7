cd ..
python test.py --data data/coco_kpts.yaml --img 960 --conf 0.001 --iou 0.65 --weights local_files_pose/yolov7-w6-pose.pt --kpt-label
