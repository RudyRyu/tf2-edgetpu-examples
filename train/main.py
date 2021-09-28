import argparse
import json
import os

import tensorflow as tf
from tensorflow.python.platform.tf_logging import error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
# import numpy as np
# import cv2

import model_utils.custom_builder
from model_utils.export_tflite_graph import export_tflite_graph
# import train.config
from train.input_pipeline import generate_tfdataset
from train.custom_model import CustomDetectorModel
from train.custom_callback import LogCallback, DetectorCheckpoint


# config = train.config.config

argparser = argparse.ArgumentParser(description='config')
argparser.add_argument(
    '-c',
    '--conf',
    required=True,
    default='train/config.py',
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
    config['checkpoint'])

tf.keras.backend.set_learning_phase(True)

batch_size = config['batch_size']
learning_rate = config['learning_rate']

if config['optimizer'] == 'SGD':
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate,
        momentum=0.9,
        nesterov=True)
else:
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

train_ds = generate_tfdataset(
    config['train_tfrecord'],
    batch_size,
    config['input_shape'][:2],
    augmentation=False, 
    normalization=True)

test_ds = generate_tfdataset(
    config['test_tfrecord'],
    batch_size,
    config['input_shape'][:2],
    augmentation=False, 
    normalization=True)

custom_model = CustomDetectorModel(
    detection_model,
    config['input_shape'],
    config['num_classes'],
    config['num_grad_accum'])
custom_model.compile(optimizer=optimizer, run_eagerly=True)

checkpoint_dir = './checkpoints/{}/best'.format(config['model_name'])
callbacks = [
    DetectorCheckpoint(detection_model, 
        monitor='val_loss', checkpoint_dir=checkpoint_dir),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, mode='min', patience=5, 
        min_lr=1e-5, verbose=1),
    LogCallback('./logs/'+config['model_name']),
    EarlyStopping(monitor='val_loss', mode='min', patience=15, 
        restore_best_weights=True)]

meta_info_path = './checkpoints/{}'.format(config['model_name'])
try:
    os.makedirs(meta_info_path, exist_ok=True)
    with open(meta_info_path+'/meta_info.config', 'w') as f:
        f.write('model{'+str(model_config)+'}')
except OSError:
    print("Error: Cannot create the directory {}".format(meta_info_path))

try:
    custom_model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=config['epoch'],
        callbacks=callbacks)
except (Exception, KeyboardInterrupt) as e:
    print()
    print('============================================')
    print('Training is canceled.')
    print(e)
    print('============================================')
    print()
else:
    export_tflite_graph(meta_info_path+'/meta_info.config', meta_info_path)

# with open('lp_test2.jpg', 'rb') as f:
#     jpeg_binary = f.read()
# image_ori = tf.io.decode_image(jpeg_binary, channels=3, expand_animations=False)
# image = tf.cast(image_ori, tf.float32)
# image = (2. / 255.) * image - 1.
# image = tf.expand_dims(image, 0)
# image = tf.image.resize(image, (512, 512))
# shapes = [[512, 512, 3]]
# prediction_dict = detection_model.predict(image, shapes)
# detections = detection_model.postprocess(prediction_dict, shapes)
# print(detections['detection_scores'][0].numpy())
# img = image_ori.numpy()
# for score, box in zip(detections['detection_scores'][0].numpy(), detections['detection_boxes'][0].numpy()):
#     if score > 0.4:
#         print(box)
#         h = img.shape[0]
#         w = img.shape[1]
#         box[0] = int(box[0] * h)
#         box[1] = int(box[1] * w)
#         box[2] = int(box[2] * h)
#         box[3] = int(box[3] * w)
#         box = box.astype(np.int32)
#         print(box)
#         img = cv2.rectangle(img, (box[1], box[0]), (box[3],box[2]), (0, 0, 255), 2)

# cv2.imwrite('{}.jpg'.format('result'), img)
