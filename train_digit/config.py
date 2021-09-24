config = {
    'model_name': 'MobileNetV2_SSD_text_seg',
    # 'model_name': 'MobileNetV2_SSD_640x640_lp_mindepth20',

    'checkpoint': '',#'checkpoints/EfficientDet_D0_SSD_640x640_lp',

    'batch_size' : 512,
    'num_grad_accum': 1,
    'epoch' : 100,
    #
    # Shape order is [Height, Width, Channel].
    #
    'input_shape' : [64, 128, 3],
    'num_classes': 17,
    #'input_shape' : [640, 640, 3],
    #'num_classes': 1,

    #
    # Choose one of below:
    # 1. MobileNetV2_SSD
    # 2. MobileNetV2_FPN_SSD
    # 3. EfficientDet_D0_SSD
    # 4. ResNet50V1_FPN_SSD
    #
    'model_type': 'MobileNetV2_SSD',

    'meta_info':{
        'matched_threshold': 0.5,
        'unmatched_threshold': 0.5,
        'base_anchor_height': 64,
        'base_anchor_width': 128,

        # bifpn is only for efficient det arch.
        'bifpn':{
            'num_iterations': 3,
            'num_filters': 64
        },
    },


    #
    # Choose one of below:
    #  1. Adam
    #  2. SGD with momentum=0.9 and nesterov=True
    #
    'optimizer' : 'Adam',

    #
    # initial learning rate.
    #
    'learning_rate' : 1e-4,

    # 'train_file': 'lp_train.tfrecord',
    # 'test_file': 'lp_valid.tfrecord',
    'train_file': 'txt_seg_train.tfrecord',
    'test_file': 'txt_seg_test.tfrecord',
}
