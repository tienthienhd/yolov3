import yolov3

model = yolov3.yolov3_net("../cfg/yolov3.cfg", (416, 416, 3), 80)
