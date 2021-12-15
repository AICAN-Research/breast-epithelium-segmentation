from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

import tensorflow as tf
print(tf.constant('OK!'))

import matplotlib.pyplot as plt
import numpy as np
f = plt.figure()
plt.imshow(np.random.rand(32, 32))
plt.show()
print('OK')





