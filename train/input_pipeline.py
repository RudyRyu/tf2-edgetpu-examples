import tensorflow as tf
import tensorflow_addons as tfa
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

import augmentation as aug

TF_AUTOTUNE = tf.data.AUTOTUNE

def generate_tfdataset(tfrecord_path, batch_size, img_shape,
                       augmentation_list=False, normalization=False):

    def _read_tfrecord(serialized):
        description = {
            'jpeg': tf.io.FixedLenFeature([], tf.string),
            'label_list': tf.io.FixedLenSequenceFeature([], tf.int64, True),
            'box_list': tf.io.FixedLenSequenceFeature([], tf.float32, True),
        }
        example = tf.io.parse_single_example(serialized, description)
        image = tf.io.decode_image(example['jpeg'], channels=3, expand_animations=False)
        image = tf.cast(image, tf.float32)
        label = example['label_list']
        box = tf.reshape(example['box_list'], (-1, 4))
        x1 = tf.reshape(box[..., 0], (-1, 1))
        y1 = tf.reshape(box[..., 1], (-1, 1))
        x2 = tf.reshape(box[..., 2], (-1, 1))
        y2 = tf.reshape(box[..., 3], (-1, 1))
        box = tf.concat([y1, x1, y2, x2], -1)
        image = tf.image.resize(image, img_shape)
        return image, label, box

    def _preprocess_image(image, label_list, box_list,
                          augmentation_list=False, normalization=False):
        image = (2.0 / 255.0) * image - 1.0
        return image, label_list, box_list

    ds = tf.data.TFRecordDataset(tfrecord_path)

    ds = (
        ds
        .map(_read_tfrecord, num_parallel_calls=TF_AUTOTUNE)
        .cache()
        .shuffle(buffer_size=tf.data.experimental.cardinality(ds).numpy())
        .map(lambda image, label_list, box_list: _preprocess_image(
            image, label_list, box_list,
            augmentation_list=augmentation_list, 
            normalization=normalization
        ), num_parallel_calls=TF_AUTOTUNE)
        .prefetch(TF_AUTOTUNE)
    )

    return ds
