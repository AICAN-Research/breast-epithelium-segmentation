import tensorflow as tf

# Augmentations
def random_brightness(x):  # look at second input parameter to random_brightness. How to deal with normalized (?)
    nbr = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    x = tf.cond(nbr < 0.5, lambda: x, lambda: tf.image.random_brightness(x, 0.1))
    return x

def random_fliplr(x, y):
    nbr = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    x, y = tf.cond(nbr < 0.5, lambda: (x, y), lambda: (tf.image.flip_left_right(x), tf.image.flip_left_right(y)))
    return x, y

def random_flipud(x, y):
    nbr = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    x, y = tf.cond(nbr < 0.5, lambda: (x, y), lambda: (tf.image.flip_up_down(x), tf.image.flip_up_down(y)))
    return x, y

def random_rot90(x ,y):  # not 90 degrees when k = 1, why??????
    nbr = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    print("nbr", nbr)
    nbr_rot = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    print("nbr rot", nbr_rot)
    x, y = tf.cond(nbr < 0.5, lambda: (x, y), lambda: (tf.image.rot90(x, k=nbr_rot), tf.image.rot90(y, k=nbr_rot)))
    return x, y