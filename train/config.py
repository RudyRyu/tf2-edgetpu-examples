
#
# for text detection
#
config = {
    'model_name': 'ShDigit_v1_SSD_text_seg',

    'checkpoint': '',

    'batch_size' : 256,
    'num_grad_accum': 1,
    'epoch' : 100,
    #
    # Shape order is [Height, Width, Channel].
    #
    'input_shape' : [64, 128, 3],
    'num_classes': 17,

    #
    # Choose one of below:
    # 1. MobileNetV2_SSD
    # 2. MobileNetV2_FPN_SSD
    # 3. EfficientDet_D0_SSD
    # 4. ResNet50V1_FPN_SSD
    #
    'model_type': 'MobileNetV2_SSD',

    'meta_info':{
        #
        # If an empty string, it is built based on 'model_type'.
        # It is used for a custom feature extractor.
        #
        'feature_extractor': 'ssd_digit_sh_keras',

        'matched_threshold': 0.5,
        'unmatched_threshold': 0.5,
        'base_anchor_height': 1.0,
        'base_anchor_width': 1.0,
        'num_layers': 2,

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

    'train_file': 'digit_train.tfrecord',
    'test_file': 'digit_valid.tfrecord',
}


# #
# # for license plate detection
# #
# config__ = {
#     'model_name': 'MobileNetV2_license_SSD_640x640',

#     'checkpoint': '',#'checkpoints/EfficientDet_D0_SSD_640x640_lp',

#     'batch_size' : 16,
#     'num_grad_accum': 16,
#     'epoch' : 50,
#     #
#     # Shape order is [Height, Width, Channel].
#     #
#     'input_shape' : [640, 640, 3],
#     'num_classes': 1,

#     #
#     # Choose one of below:
#     # 1. MobileNetV2_SSD
#     # 2. MobileNetV2_FPN_SSD
#     # 3. EfficientDet_D0_SSD
#     # 4. ResNet50V1_FPN_SSD
#     #
#     'model_type': 'MobileNetV2_SSD',

#     'meta_info':{
#         #
#         # If an empty string, it is built based on 'model_type'.
#         # It is used for a custom feature extractor.
#         #
#         'feature_extractor': '',

#         'matched_threshold': 0.5,
#         'unmatched_threshold': 0.5,
#         'base_anchor_height': 1.0,
#         'base_anchor_width': 1.0,
#         'num_layers': 6,

#         # bifpn is only for efficient det arch.
#         'bifpn':{
#             'num_iterations': 3,
#             'num_filters': 64
#         },
#     },

#     #
#     # Choose one of below:
#     #  1. Adam
#     #  2. SGD with momentum=0.9 and nesterov=True
#     #
#     'optimizer' : 'Adam',

#     #
#     # initial learning rate.
#     #
#     'learning_rate' : 1e-4,

#     'train_file': 'lp_train.tfrecord',
#     'test_file': 'lp_valid.tfrecord',
# }
