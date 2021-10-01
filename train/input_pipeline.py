import cv2
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

import augmentation as aug

TF_AUTOTUNE = tf.data.AUTOTUNE

def get_tfdataset_length(dataset: tf.data.TFRecordDataset):
    data_len = 0
    for _ in dataset:
        data_len += 1

    return data_len

def generate_tfdataset(tfrecord_path, batch_size, image_size_hw,
                       augmentation=False, normalization=False):

    def _parse_tfrecord(serialized):
        description = {
            'jpeg': tf.io.FixedLenFeature([], tf.string),
            'label_list': tf.io.FixedLenSequenceFeature([], tf.int64, True),
            'box_list': tf.io.FixedLenSequenceFeature([], tf.float32, True),
        }
        example = tf.io.parse_single_example(serialized, description)
        image = tf.io.decode_image(example['jpeg'], channels=3, 
            expand_animations=False)
        image = tf.cast(image, tf.float32)
        label_list = example['label_list']
        box_list = tf.reshape(example['box_list'], (-1, 4))
        x1 = tf.reshape(box_list[..., 0], (-1, 1))
        y1 = tf.reshape(box_list[..., 1], (-1, 1))
        x2 = tf.reshape(box_list[..., 2], (-1, 1))
        y2 = tf.reshape(box_list[..., 3], (-1, 1))
        box_list = tf.concat([y1, x1, y2, x2], -1)
        image = tf.image.resize(image, image_size_hw)

        return image, label_list, box_list
    
    def _preprocess(image, label_list, box_list):
        
        image_shape = image.shape
        box_shape = box_list.shape

        if augmentation:
            image, box_list = tf.numpy_function(
                func=aug.augmentation_pipeline,
                inp=[image, box_list],
                Tout=[tf.float32, tf.float32])

            image = tf.ensure_shape(image, image_shape)
            box_list  = tf.ensure_shape(box_list, box_shape)

        if normalization:
            image = (2.0 / 255.0) * image - 1.0

        return image, label_list, box_list

    ds = tf.data.TFRecordDataset(tfrecord_path)
    
    ds = (
        ds
        .map(_parse_tfrecord, num_parallel_calls=TF_AUTOTUNE)
        .cache()
        .shuffle(buffer_size=get_tfdataset_length(ds))
        .map(lambda image, label_list, box_list: _preprocess(
                image, label_list, box_list), 
            num_parallel_calls=TF_AUTOTUNE)
        .apply(tf.data.experimental.dense_to_ragged_batch(batch_size,
            drop_remainder=True))
        .prefetch(TF_AUTOTUNE)
    )

    return ds

if __name__ == '__main__':
    ds = generate_tfdataset('data/digit_valid.tfrecord', 32, [64,128],
        augmentation=False, normalization=False)

    for images, label_lists, box_lists in ds.take(5):
        for image, label_list, box_list in zip(images, label_lists, box_lists):
            np_image = image.numpy().astype(np.uint8)
            np_image_bgr = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

            print('label', label_list.numpy())
            print('box', box_list.numpy())
            print()
            cv2.imshow('img', np_image_bgr)
            cv2.waitKey()


    # for images, label_lists, box_lists in ds.take(1):
    #     for image, label_list, box_list in zip(images, label_lists, box_lists):
    #         cv2.imshow(image.numpy())
    #         cv2.waitKey()
