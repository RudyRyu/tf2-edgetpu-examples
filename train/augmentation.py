import tensorflow as tf
import tensorflow_addons as tfa

def random_flip(x: tf.Tensor):
    return tf.image.random_flip_left_right(x)

def gray(x):
        return tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(x))

def random_color(x: tf.Tensor):
    x = tf.image.random_hue(x, 0.2)
    x = tf.image.random_brightness(x, 0.2)
    x = tf.image.random_contrast(x, 0.8, 1.2)
    return x

def blur(x):
    choice = tf.random.uniform([], 0, 1, dtype=tf.float32)
    def gfilter(x):
        return tfa.image.gaussian_filter2d(x, [5, 5], 1.0, 'REFLECT', 0)


    def mfilter(x):
        return tfa.image.median_filter2d(x, [5, 5], 'REFLECT', 0)

    return tf.cond(choice > 0.5, lambda: gfilter(x), lambda: mfilter(x))