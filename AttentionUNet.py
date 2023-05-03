"""
From:
https://github.com/andreped/H2G-Net/blob/main/src/architectures/AttentionUNet.py
the AGU-Net was again base on "Meningioma segmentation in T1-weighted MRI leveraging global context
and attention mechanisms" by Bouget et al (2021)
Some minor changed made (and removed unused code)
"""

from tensorflow.keras.layers import Input, Convolution2D, MaxPooling2D, SpatialDropout2D, \
    Activation, AveragePooling2D, BatchNormalization, TimeDistributed, Concatenate, Conv2DTranspose, \
    UpSampling2D, multiply, Reshape, Layer
from tensorflow.keras.models import Model
import tensorflow as tf
from gradient_accumulator import AccumBatchNormalization


def convolution_block(x, nr_of_convolutions, accum_steps=None, use_bn=False, spatial_dropout=None, renorm=False,
                      use_grad_accum=False):
    for i in range(2):
        x = Convolution2D(nr_of_convolutions, 3, padding='same')(x)
        if use_bn:
            x = BatchNormalization(renorm=renorm)(x)
        if use_grad_accum:
            x = AccumBatchNormalization(accum_steps=accum_steps)(x)
        x = Activation('relu')(x)
        if spatial_dropout:
            x = SpatialDropout2D(spatial_dropout)(x)

    return x


def attention_block(g, x, nr_of_convolutions, accum_steps=None, renorm=False, use_grad_accum=False):
    """
    Taken from https://github.com/LeeJunHyun/Image_Segmentation
    """
    g1 = Convolution2D(nr_of_convolutions, kernel_size=1, strides=1, padding='same', use_bias=True)(g)
    if use_grad_accum:
        g1 = AccumBatchNormalization(accum_steps=accum_steps)(g1)

        x1 = Convolution2D(nr_of_convolutions, kernel_size=1, strides=1, padding='same', use_bias=True)(x)
        x1 = AccumBatchNormalization(accum_steps=accum_steps)(x1)

        psi = Concatenate()([g1, x1])
        psi = Activation(activation='relu')(psi)
        psi = Convolution2D(1, kernel_size=1, strides=1, padding='same', use_bias=True)(psi)
        psi = AccumBatchNormalization(accum_steps=accum_steps)(psi)
        psi = Activation(activation='sigmoid')(psi)
    else:
        #g1 = BatchNormalization(renorm=renorm)(g1)

        x1 = Convolution2D(nr_of_convolutions, kernel_size=1, strides=1, padding='same', use_bias=True)(x)
        #x1 = BatchNormalization(renorm=renorm)(x1)

        psi = Concatenate()([g1, x1])
        psi = Activation(activation='relu')(psi)
        psi = Convolution2D(1, kernel_size=1, strides=1, padding='same', use_bias=True)(psi)
        #psi = BatchNormalization(renorm=renorm)(psi)
        psi = Activation(activation='sigmoid')(psi)

    return multiply([x, psi])


def encoder_block(x, nr_of_convolutions, accum_steps=None, use_bn=False, spatial_dropout=None, renorm=False,
                  use_grad_accum=False):
    x_before_downsampling = convolution_block(x, nr_of_convolutions, accum_steps=accum_steps, use_bn=use_bn,
                                              spatial_dropout=spatial_dropout, renorm=renorm,
                                              use_grad_accum=use_grad_accum)
    downsample = [2, 2]
    for i in range(1, 3):
        if x.shape[i] <= 3:
            downsample[i - 1] = 1

    x = MaxPooling2D(downsample)(x_before_downsampling)

    return x, x_before_downsampling


def encoder_block_pyramid(x, input_ds, nr_of_convolutions, accum_steps=None, use_bn=False, spatial_dropout=None,
                          renorm=False, use_grad_accum=False):
    # pyramid_conv = convolution_block(input_ds, nr_of_convolutions, use_bn, spatial_dropout)
    pyramid_conv = Convolution2D(filters=nr_of_convolutions, kernel_size=(3, 3), padding='same', activation='relu')(
        input_ds)
    x = Concatenate(axis=-1)([pyramid_conv, x])
    x_before_downsampling = convolution_block(x, nr_of_convolutions, accum_steps=accum_steps, use_bn=use_bn,
                                              spatial_dropout=spatial_dropout, renorm=renorm,
                                              use_grad_accum=use_grad_accum)
    downsample = [2, 2]
    for i in range(1, 3):
        if x.shape[i] <= 4:
            downsample[i - 1] = 1

    x = MaxPooling2D(downsample)(x_before_downsampling)

    return x, x_before_downsampling


def decoder_block(x, cross_over_connection, nr_of_convolutions, accum_steps=None, use_bn=False, spatial_dropout=None,
                  renorm=False, use_grad_accum=False):
    x = UpSampling2D((2, 2))(x)  # See if this helps with checkerboard pattern sometimes seen
    if use_bn:
        x = BatchNormalization(renorm=renorm)(x)
    if use_grad_accum:
        x = AccumBatchNormalization(accum_steps=accum_steps)(x)
    x = Activation('relu')(x)
    attention = attention_block(g=x, x=cross_over_connection, nr_of_convolutions=int(nr_of_convolutions / 2),
                                accum_steps=accum_steps, renorm=renorm, use_grad_accum=use_grad_accum)
    x = Concatenate()([x, attention])
    x = convolution_block(x, nr_of_convolutions, use_bn, spatial_dropout, renorm=renorm)

    return x


class AttentionUnet:
    def __init__(self, input_shape, nb_classes, encoder_spatial_dropout, decoder_spatial_dropout, accum_steps,
                 deep_supervision=False, input_pyramid=False, grad_accum=False, encoder_use_bn=False,
                 decoder_use_bn=False):
        if len(input_shape) != 3 and len(input_shape) != 4:
            raise ValueError('Input shape must have 3 or 4 dimensions')
        if nb_classes <= 1:
            raise ValueError('Segmentation classes must be > 1')
        self.dims = 2
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.deep_supervision = deep_supervision
        self.input_pyramid = input_pyramid
        self.convolutions = None
        self.encoder_use_bn = encoder_use_bn
        self.decoder_use_bn = decoder_use_bn
        self.encoder_spatial_dropout = encoder_spatial_dropout  # used to be None
        self.decoder_spatial_dropout = decoder_spatial_dropout  # used to be None
        self.renorm = False
        self.grad_accum = grad_accum
        self.accum_steps = accum_steps

    def set_renorm(self, value):
        self.renorm = value

    def set_convolutions(self, convolutions):
        self.convolutions = convolutions


    def create(self):
        """
        Create model and return it

        :return: keras model
        """

        input_layer = Input(shape=self.input_shape)
        x = input_layer
        connection = []

        if self.input_pyramid:
            scaled_input = []
            scaled_input.append(x)
            for i, nbc in enumerate(self.convolutions[:-1]):
                ds_input = AveragePooling2D(pool_size=(2, 2))(scaled_input[i])
                scaled_input.append(ds_input)

        for i, nbc in enumerate(self.convolutions[:-1]):
            if not self.input_pyramid or (i == 0):
                x, x_before_ds = encoder_block(x, nbc, accum_steps=self.accum_steps, use_bn=self.encoder_use_bn,
                                               spatial_dropout=self.encoder_spatial_dropout, renorm=self.renorm,
                                               use_grad_accum=self.grad_accum)
            else:
                x, x_before_ds = encoder_block_pyramid(x, scaled_input[i], nbc, accum_steps=self.accum_steps,
                                                       use_bn=self.encoder_use_bn,
                                                       spatial_dropout=self.encoder_spatial_dropout, renorm=self.renorm,
                                                       use_grad_accum=self.grad_accum)
            connection.insert(0, x_before_ds)  # Append in reverse order for easier use in the next block

        x = convolution_block(x, self.convolutions[-1], accum_steps=self.accum_steps,
                              use_bn=self.encoder_use_bn, spatial_dropout=self.encoder_spatial_dropout,
                              renorm=self.renorm, use_grad_accum=self.grad_accum)
        connection.insert(0, x)

        inverse_conv = self.convolutions[::-1]
        inverse_conv = inverse_conv[1:]
        decoded_layers = []
        # @TODO. Should Attention Gating be done over the last feature map (i.e. image at the highest resolution)?
        # Some papers say they don't because the feature map does not represent the data in a high dimensional space.
        for i, nbc in enumerate(inverse_conv):
            x = decoder_block(x, connection[i + 1], nbc, self.accum_steps, use_bn=self.decoder_use_bn,
                              spatial_dropout=self.decoder_spatial_dropout, renorm=self.renorm,
                              use_grad_accum=self.grad_accum)
            decoded_layers.append(x)

        if not self.deep_supervision:
            # Final activation layer
            x = Convolution2D(self.nb_classes, 1, activation='softmax', dtype=tf.float32)(x)
        else:
            recons_list = []
            for i, lay in enumerate(decoded_layers):
                x = Convolution2D(self.nb_classes, 1, activation='softmax', dtype=tf.float32)(lay)
                recons_list.append(x)
            x = recons_list[::-1]

        return Model(inputs=input_layer, outputs=x)

