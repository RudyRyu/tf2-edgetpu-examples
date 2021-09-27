import os

from object_detection.utils import config_util
from model_utils import ssd_model_builder

import tensorflow.compat.v2 as tf


def build(base_config_path, input_shape, num_classes, 
          meta_info, checkpoint):
    base_configs = config_util.get_configs_from_pipeline_file(base_config_path)
    
    # base config
    model_config = base_configs['model']

    # custom config
    model_config.ssd.num_classes = num_classes
    model_config.ssd.freeze_batchnorm = False
    model_config.ssd.image_resizer.fixed_shape_resizer.height = input_shape[0]
    model_config.ssd.image_resizer.fixed_shape_resizer.width = input_shape[1]
    model_config.ssd.matcher.argmax_matcher.matched_threshold = meta_info['matched_threshold']
    model_config.ssd.matcher.argmax_matcher.unmatched_threshold = meta_info['unmatched_threshold']
    model_config.ssd.anchor_generator.ssd_anchor_generator.num_layers = meta_info['num_layers']
    if meta_info['feature_extractor']:
        model_config.ssd.feature_extractor.type = meta_info['feature_extractor']
    model_config.ssd.feature_extractor.num_layers = meta_info['num_layers']
    detection_model = ssd_model_builder.build(model_config=model_config, is_training=True)

    # if os.path.exists(checkpoint['dir']):

    #     ckpt = tf.train.Checkpoint(model=detection_model)
    #     manager = tf.train.CheckpointManager(
    #         ckpt, checkpoint, max_to_keep=1)
        
    #     checkpoint_file_path = os.path.join(
    #         checkpoint['dir'], checkpoint['name'])

    #     if os.path.exists(checkpoint_file_path):
    #         ckpt.restore(checkpoint_file_path).expect_partial()

    #     if manager.latest_checkpoint is not None:
    #         ckpt.restore(manager.latest_checkpoint)
    #     else:
    #         print()
    #         print('============================================')
    #         print('You gave checkpoint path, but not exists. -> {}'.format(checkpoint['dir']))
    #         print('============================================')
    #         print()

    # # Run model through a dummy image so that variables are created
    # image, shapes = detection_model.preprocess(tf.zeros([1] + input_shape))
    # prediction_dict = detection_model.predict(image, shapes)
    # _ = detection_model.postprocess(prediction_dict, shapes)
    # print('Weights restored!')

    return detection_model, model_config
