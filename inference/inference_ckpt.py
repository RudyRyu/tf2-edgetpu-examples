import argparse
import json
import os
from pprint import pprint

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.platform.tf_logging import error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from object_detection.utils import visualization_utils as viz_utils
import model_utils.custom_builder


def detect(detection_model, np_image, input_shape_wh):
    np_image = cv2.resize(np_image, input_shape_wh)
    tf_image = tf.convert_to_tensor(np_image, dtype=tf.float32)
    tf_image = tf.expand_dims(tf_image, axis=0)

    image, shapes = detection_model.preprocess(tf_image)

    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    boxes = detections['detection_boxes'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(np.uint32)
    scores = detections['detection_scores'][0].numpy()

    return boxes, classes, scores


argparser = argparse.ArgumentParser(description='config')
argparser.add_argument(
    '-c',
    '--conf',
    required=True,
    help='path to a configuration file')

args = argparser.parse_args()

with open(args.conf) as config_buffer:
    config = json.loads(config_buffer.read())

tf.keras.backend.clear_session()

detection_model, model_config = model_utils.custom_builder.build(
    config['base_config_path'],
    config['input_shape'],
    config['num_classes'],
    config['meta_info'],
    config['checkpoint_path'])

input_shape_wh = (512, 384)
cap = cv2.VideoCapture(config['video_path'])
cap.set(3, input_shape_wh[0])
cap.set(4, input_shape_wh[1])


while True:
    _, image = cap.read()
    if image is None:
        continue

    image = cv2.resize(image, input_shape_wh)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    boxes, classes, scores = detect(detection_model, image, input_shape_wh)

    for box, class_id, score in zip(boxes, classes, scores):

        if score < 0.4:
            continue
        
        x1 = int(box[1] * input_shape_wh[0])
        y1 = int(box[0] * input_shape_wh[1])
        x2 = int(box[3] * input_shape_wh[0])
        y2 = int(box[2] * input_shape_wh[1])

        pprint(score)
        pprint(box)
        pprint(class_id)
        print(x1,y1,x2,y2)

        image = cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2)
        
    cv2.imshow('res', image)
    cv2.waitKey(1)
        


