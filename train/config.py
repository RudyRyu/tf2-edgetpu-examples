config = {
    'model_name': 'ShDigit_v1_SSD_text_seg',

    'base_config_path': 'model_utils/builder_configs/ssd_mobilenet_v2.config',

    # input shape == [height, width, channel]
    'input_shape' : [64, 128, 3],

    'num_classes': 17,
    
    # total batch size == batch_size * num_grad_accum
    'batch_size' : 256,
    'num_grad_accum': 1,

    'epoch' : 100,

    # initial learning rate
    'learning_rate' : 1e-4,

    # 1. Adam
    # 2. SGD (with momentum=0.9 and nesterov=True)
    'optimizer' : 'Adam',

    'meta_info':{
        'feature_extractor': 'ssd_digit_sh_keras',
        'num_layers': 2,

        'matched_threshold': 0.5,
        'unmatched_threshold': 0.5,
        'base_anchor_height': 1.0,
        'base_anchor_width': 1.0,
    },

    # Restore weights from checkpoint
    'checkpoint': {
        'dir': 'checkpoints/ShDigit_v1_SSD_text_seg/',
        'name': 'best-50'
    },

    'train_tfrecord': 'digit_train.tfrecord',
    'test_tfrecord': 'digit_valid.tfrecord',
}