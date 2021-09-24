import tensorflow as tf

def get_tfdataset(tfrecord_file, batch_size, image_shape_hwc):
    
    def _map_read_tfrecord(serialized):
        description = {
            'jpeg': tf.io.FixedLenFeature((), tf.string),
            'label': tf.io.FixedLenFeature((), tf.int64),
            'x1': tf.io.FixedLenFeature((), tf.float32),
            'y1': tf.io.FixedLenFeature((), tf.float32),
            'x2': tf.io.FixedLenFeature((), tf.float32),
            'y2': tf.io.FixedLenFeature((), tf.float32)
        }
        example = tf.io.parse_single_example(serialized, description)
        # image = tf.io.decode_jpeg(example['jpeg'], channels=3)
        image = tf.io.decode_image(
            example['jpeg'], channels=3, expand_animations=False)
        image = tf.cast(image, tf.float32)
        label = example['label']
        box = [example['y1'], example['x1'], example['y2'], example['x2']]
        image = tf.image.resize(image, image_shape_hwc)
        return image, label, box

    def _map_eager_decorator(func):
        def wrapper(images, labels, boxes):
            return tf.py_function(
                func,
                inp=(images, labels, boxes),
                Tout=(images.dtype, labels.dtype, boxes.dtype)
            )
        return wrapper

    def _map_preprocess_images(imgs, labels, boxes):
        # imgs = (2.0 / 255.0) * imgs - 1.0
        imgs = imgs / 255.0
        return imgs, labels, boxes

    ds = tf.data.TFRecordDataset(tfrecord_file)

    ds = ds.map(
        _map_read_tfrecord, 
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    