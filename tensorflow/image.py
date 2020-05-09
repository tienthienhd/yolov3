import time

import tensorflow as tf
from utils import load_class_names, output_boxes, draw_output, resize_image
import cv2
import numpy as np
from yolov3 import yolov3_net

# physical_devices = tf.config.experimental.list_logical_devices("GPU")
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

model_size = (416, 416, 3)
num_classes = 80
class_names_file = '../data/coco.names'
max_output_size = 40
max_output_size_per_class = 20
iou_threshold = 0.5
confidence_threshold = 0.5

cfg_file = '../cfg/yolov3.cfg'
weights_file = '../weights/yolov3_weights.tf'

img_path = '../images/dog-cycle-car.png'
# img_path = '/home/thiennt/Desktop/selfie_2020-03-02_10-39-41_crop.jpg'
# img_path = '/home/thiennt/Desktop/t.jpeg'


def main():
    model = yolov3_net(cfg_file, num_classes)
    model.load_weights(weights_file)

    class_names = load_class_names(class_names_file)

    image = cv2.imread(img_path)
    image = tf.expand_dims(image, 0)
    resized_frame = resize_image(image, (model_size[0], model_size[1]))
    start_time = time.time()
    pred = model.predict(resized_frame, steps=1)
    print("Time inference: ", time.time() - start_time)
    boxes, scores, classes, nums = output_boxes(pred, model_size, max_output_size, max_output_size_per_class,
                                                iou_threshold, confidence_threshold)

    image = np.squeeze(image)
    img = draw_output(image, boxes, scores, classes, nums, class_names)

    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    win_name = "Image detection"
    cv2.imshow(win_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
