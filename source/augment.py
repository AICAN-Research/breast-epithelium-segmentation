import tensorflow as tf
import tensorflow_addons as tfa
import random


# Augmentations
def random_brightness(x,
                      brightness=25):  # look at second input parameter to random_brightness. How to deal with normalized (?)
    nbr = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    x = tf.cond(nbr < 0.5, lambda: x, lambda: tf.clip_by_value(tf.image.random_brightness(x, brightness), 0, 1))
    return x


def random_fliplr(x, y):
    nbr = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    x, y = tf.cond(nbr < 0.5, lambda: (x, y), lambda: (tf.image.flip_left_right(x), tf.image.flip_left_right(y)))
    return x, y


def random_flipud(x, y):
    nbr = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    x, y = tf.cond(nbr < 0.5, lambda: (x, y), lambda: (tf.image.flip_up_down(x), tf.image.flip_up_down(y)))
    return x, y


def random_rot90(x, y):  # not 90 degrees when k = 1, why??????
    nbr = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    nbr_rot = tf.random.uniform(shape=[], minval=1, maxval=4, dtype=tf.int32)
    x, y = tf.cond(nbr < 0.5, lambda: (x, y), lambda: (tf.image.rot90(x, k=nbr_rot), tf.image.rot90(y, k=nbr_rot)))
    return x, y


def random_hue(x, max_delta):
    nbr = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    x = tf.cond(nbr < 0.5, lambda: x, lambda: tf.clip_by_value(tf.image.random_hue(x, max_delta=max_delta), 0, 1))
    return x


def random_saturation(x, saturation):
    nbr = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    x = tf.cond(nbr < 0.5, lambda: x, lambda: tf.clip_by_value(
        tf.image.random_saturation(x, lower=1 - saturation, upper=1 + saturation), 0, 1))
    return x


@tf.function
def random_blur(x):
    nbr = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    sigma_ = random.uniform(1, 3)  # @TODO: find a way to avoid numpy, will switch to cpu
    x = tf.cond(nbr < 0.9, lambda: x, lambda: tf.clip_by_value(tfa.image.gaussian_filter2d(
        x, filter_shape=15, sigma=sigma_), 0, 1))   # @TODO: get tf warning sometimes, find out why
    return x


#  @TODO: consider adding additive gaussian noise too


def random_contrast(x, low, up):
    nbr = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    x = tf.cond(nbr < 0.5, lambda: x, lambda: tf.clip_by_value(
        tf.image.random_contrast(x, lower=low, upper=up), 0, 1))
    return x


# convenience
def gt_shift(z, shift_x, shift_y):
    # need to handle one-hots in a specific way, when shifting
    tmp1 = tfa.image.translate(z[..., 0:1], (shift_x, shift_y), fill_value=1.0)  # background special case!!
    tmp2 = tfa.image.translate(z[..., 1:], (shift_x, shift_y), fill_value=0.0)  # all other classes, do same padding
    z = tf.concat([tmp1, tmp2], axis=-1)
    return z


def random_shift(x, y, translate=50):
    choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    shift_x = tf.random.uniform(shape=[], minval=-translate, maxval=translate, dtype=tf.int32)
    shift_y = tf.random.uniform(shape=[], minval=-translate, maxval=translate, dtype=tf.int32)

    x, y = tf.cond(choice < 0.5, lambda: (x, y), lambda: (
        tfa.image.translate(x, (shift_x, shift_y)),
        gt_shift(y, shift_x, shift_y),
    )
                   )
    return x, y
