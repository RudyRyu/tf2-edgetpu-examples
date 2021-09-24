from pprint import pprint

import tensorflow as tf

from google.protobuf import text_format
from object_detection.protos import pipeline_pb2
from model_utils import export_tflite_graph_lib_tf2

import train

config = train.config.config
pprint(config)


def export_tflite_graph(pipeline_config_path, checkpoint_path, 
                        checkpoint_name=''):
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    max_detections = 10
    config_override = ''
    ssd_use_regular_nms = False
    centernet_include_keypoints = False
    keypoint_label_map_path = None

    with tf.io.gfile.GFile(pipeline_config_path, 'r') as f:
        text_format.Parse(f.read(), pipeline_config)
    override_config = pipeline_pb2.TrainEvalPipelineConfig()
    text_format.Parse(config_override, override_config)
    pipeline_config.MergeFrom(override_config)

    export_tflite_graph_lib_tf2.export_tflite_model(
        pipeline_config, checkpoint_path, checkpoint_path,
        max_detections, ssd_use_regular_nms,
        centernet_include_keypoints, keypoint_label_map_path, checkpoint_name)


if __name__ == '__main__':
    meta_info_path = './checkpoints/{}'.format(config['model_name'])
    export_tflite_graph(meta_info_path+'/meta_info.config', meta_info_path)