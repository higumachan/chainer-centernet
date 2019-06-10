from functools import partial

from chainer import Chain, Sequential, ChainList
import chainer.links as L
import chainer.functions as F
from typing import List, Dict




def conv_bn_relu(kernel_size, input_dim, output_dim, stride=1, with_bn=True) -> Sequential:
    pad = (kernel_size - 1) // 2
    bn = L.BatchNormalization(output_dim) if with_bn else F.identity
    return Sequential(
        L.Convolution2D(input_dim, output_dim, kernel_size, stride, pad, nobias=(not with_bn)),
        bn,
        F.relu,
    )
