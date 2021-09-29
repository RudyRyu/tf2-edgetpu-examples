import time

import cv2
import numpy as np
from numpy.lib.function_base import flip
import tensorflow as tf
import tensorflow_addons as tfa
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


def tf_random_flip(x: tf.Tensor):
    return tf.image.random_flip_left_right(x)

def tf_gray(x):
        return tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(x))

def tf_random_color(x: tf.Tensor):
    x = tf.image.random_hue(x, 0.2)
    x = tf.image.random_brightness(x, 0.2)
    x = tf.image.random_contrast(x, 0.8, 1.2)
    return x

def tf_blur(x):
    choice = tf.random.uniform([], 0, 1, dtype=tf.float32)
    def gfilter(x):
        return tfa.image.gaussian_filter2d(x, [5, 5], 1.0, 'REFLECT', 0)


    def mfilter(x):
        return tfa.image.median_filter2d(x, [5, 5], 'REFLECT', 0)

    return tf.cond(choice > 0.5, lambda: gfilter(x), lambda: mfilter(x))

def augmentation_pipeline(image, box_list,
                          random_geometry=True,
                          random_color=True):

    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    h,w = image.shape[:2]

    # ia.seed(int(time.time()))
    sometimes1 = lambda aug: iaa.Sometimes(0.1, aug)
    sometimes3 = lambda aug: iaa.Sometimes(0.3, aug)
    sometimes5 = lambda aug: iaa.Sometimes(0.5, aug)
    sometimes7 = lambda aug: iaa.Sometimes(0.7, aug)
    sometimes9 = lambda aug: iaa.Sometimes(0.9, aug)
    if_ = lambda tf, t, f: t if tf else f

    seq = iaa.Sequential([
        # random transform
        iaa.SomeOf((1, if_(random_geometry, 3, 0)),[
            iaa.PiecewiseAffine(scale=(0, 0.02)),
            iaa.PerspectiveTransform(scale=(0.00, 0.08), fit_output=True),
            # iaa.ElasticTransformation(alpha=(0, 1.0), sigma=0.1),
            iaa.Affine(shear=(-17,17), fit_output=True)
        ], random_order=True),

        # random color
        iaa.SomeOf((2, if_(random_color, 6, 0)),[
            iaa.LinearContrast((0.7, 1.3), per_channel=0.5),
            iaa.Dropout(p=(0, 0.01), per_channel=0.5),
            iaa.Add((-40, 40), per_channel=0.5),
            iaa.Sharpen(alpha=(0.3, 0.7), lightness=(0.75, 1.25)),
            iaa.Grayscale(alpha=(0.1, 1.0)),

            iaa.Sequential([
                 iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
                 iaa.WithChannels(0, iaa.Add((10, 100))),
                 iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")
            ]),

            # # Blur
            # iaa.SomeOf(1,[
            #     iaa.GaussianBlur(sigma=(0.1, 1.0)),
            #     iaa.AverageBlur(k=(1, 3)),
            #     iaa.MotionBlur(k=5, angle=[-45, 45]),
            #     iaa.BilateralBlur(d=(1, 7),
            #                       sigma_color=(10, 250), 
            #                       sigma_space=(10, 250))
            # ]),
        ], random_order=True),

        iaa.Fliplr(0.5),
        iaa.Resize({"height": h, "width": w})
    ])
    
    bbox_list = []
    for box in box_list:
        bbox_list.append(
            BoundingBox(
                x1=int(box[1]*w), y1=int(box[0]*h),
                x2=int(box[3]*w), y2=int(box[2]*h)))

    bbox_list_on_image = BoundingBoxesOnImage(bbox_list, shape=image.shape)

    aug_image, aug_bbox_list = seq(
        image=image, bounding_boxes=bbox_list_on_image)
        
    aug_box_list = []
    for bbox in aug_bbox_list:
        x1 = bbox.x1 / w
        y1 = bbox.y1 / h
        x2 = bbox.x2 / w
        y2 = bbox.y2 / h
        aug_box_list.append([y1,x1,y2,x2])
    np_aug_box_list = np.array(aug_box_list, np.float32)

    return aug_image.astype(np.float32), np_aug_box_list
