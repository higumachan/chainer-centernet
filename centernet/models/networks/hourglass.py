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


def fully_connected(input_dim, output_dim, with_bn=True):
    bn = L.BatchNormalization(output_dim) if with_bn else F.identity
    return Sequential(
        L.Linear(input_dim, output_dim),
        bn,
        F.relu,
    )


class Residual(Chain):
    def __init__(self, kernel_size, input_dim, output_dim, stride=1):
        super().__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(input_dim, output_dim, kernel_size, stride, (1, 1), nobias=True)
            self.bn1 = L.BatchNormalization(output_dim)
            self.conv2 = L.Convolution2D(output_dim, output_dim, kernel_size, 1, (1, 1), nobias=True)
            self.bn2 = L.BatchNormalization(output_dim)
            self.skip = Sequential(
                L.Convolution2D(input_dim, output_dim, (1, 1), stride=(stride, stride), nobias=True),
                L.BatchNormalization(output_dim)
            ) if stride != 1 or input_dim != output_dim else F.identity

    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.conv2(h)
        h = self.bn2(h)

        skip = self.skip(x)

        return F.relu(h + skip)


def make_layers(kernel, input_dim, output_dim, modules, layer=conv_bn_relu):
    head = layer(kernel, input_dim, output_dim)
    tails = layer(kernel, output_dim, output_dim).repeat(modules-1)
    return Sequential(head, tails)


def make_layers_reverse(kernel, input_dim, output_dim, modules, layer=conv_bn_relu):
    heads = layer(kernel, input_dim, input_dim).repeat(modules-1)
    tail = layer(kernel, input_dim, output_dim)

    return Sequential(heads, tail)


def make_pool_layers(dim):
    return Sequential(F.identity)


def make_unpool_layers(dim):
    return Sequential(partial(F.unpooling_2d, ksize=2, stride=2, cover_all=False))


def make_kp_layers(conv_dim, curr_dim, out_dim):
    return Sequential(
        conv_bn_relu(3, conv_dim, curr_dim, with_bn=False),
        L.Convolution2D(curr_dim, out_dim, (1, 1))
    )


def make_inter_layers(dim):
    return Residual(3, dim, dim)


def make_conv_layers(input_dim, output_dim):
    return conv_bn_relu(3, input_dim, output_dim)


class KpChain(Chain):

    #make_merge_layers = make_merge_layers

    def __init__(self, n, dims, modules,
                 layer=Residual,
                 make_up_layers=make_layers,
                 make_low_layers=make_layers,
                 make_hg_layers=make_layers,
                 make_hg_layer_reverse=make_layers_reverse,
                 make_pool_layers=make_pool_layers,
                 make_unpool_layers=make_unpool_layers,
    ):
        super().__init__()

        self.n = n

        curr_mod = modules[0]
        next_mod = modules[1]
        curr_dim = dims[0]
        next_dim = dims[1]

        with self.init_scope():
            self.up1 = make_up_layers(3, curr_dim, curr_dim, curr_mod, layer=layer)
            self.max1 = make_pool_layers(curr_dim)
            self.low1 = make_hg_layers(3, curr_dim, next_dim, curr_mod, layer=layer)
            self.low2 = KpChain(n-1, dims[1:], modules[1:], make_hg_layers=make_hg_layers) if self.n > 1 else make_low_layers(
                3, next_dim, next_dim, next_mod, layer=layer
            )
            self.low3 = make_hg_layer_reverse(3, next_dim, curr_dim, curr_mod, layer=layer)
            self.up2 = make_unpool_layers(curr_mod)

    def forward(self, x):
        up1 = self.up1(x)
        max1 = self.max1(x)
        low1 = self.low1(max1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)
        return up1 + up2


class ExKp(Chain):


    def __init__(self, n, nstack, dims, modules, heads, pre=None, conv_dim=256,
                 make_tl_layer = None,
                 make_br_layer = None,
                 make_conv_layers = make_conv_layers,
                 make_heat_layer = make_kp_layers,
                 make_tag_layers = make_kp_layers,
                 make_regr_layer = make_kp_layers,
                 make_up_layer = make_layers,
                 make_low_layer = make_layers,
                 make_hg_layers = make_layers,
                 make_hg_layer_reverse = make_layers_reverse,
                 make_pool_layers = make_pool_layers,
                 make_unpool_layers = make_unpool_layers,
                 make_inter_layer = make_inter_layers,
                 kp_layer = Residual,
    ):
        super().__init__()

        self.nstack = nstack
        self.heads = heads

        curr_dim = dims[0]

        with self.init_scope():
            self.pre = Sequential(
                conv_bn_relu(7, 3, 128, stride=2),
                Residual(3, 128, 256, stride=2)
            ) if pre is None else pre

            self.kps = ChainList(*[KpChain(n, dims, modules, make_hg_layers=make_hg_layers) for _ in range(nstack)])
            self.convs = ChainList(*[make_conv_layers(curr_dim, conv_dim) for _ in range(nstack)])
            self.inters = ChainList(*[make_inter_layer(curr_dim) for _ in range(nstack - 1)])
            self.inters_ = ChainList(
                *[Sequential(
                    L.Convolution2D(curr_dim, curr_dim, (1, 1), nobias=True),
                    L.BatchNormalization(curr_dim),
                ) for _ in range(nstack - 1)]
            )
            self.convs_ = ChainList(*[
                *[Sequential(
                    L.Convolution2D(conv_dim, curr_dim, (1, 1), nobias=True),
                    L.BatchNormalization(curr_dim),
                ) for _ in range(nstack - 1)]
            ])

            for head in heads.keys():
                if 'hm' in head:
                    c = ChainList(*[
                        make_heat_layer(conv_dim, curr_dim, heads[head]) for _ in range(nstack)
                    ])
                    self.__setattr__(head, c)
                else:
                    c = ChainList(*[
                        make_regr_layer(conv_dim, curr_dim, heads[head]) for _ in range(nstack)
                    ])
                    self.__setattr__(head, c)

    def forward(self, x):
        inter = self.pre(x)

        outs = []

        for ind in range(self.nstack):
            kp_, conv_ = self.kps[ind], self.convs[ind]
            kp = kp_(inter)
            conv = conv_(kp)

            out = {}
            for head in self.heads:
                layer = self.__getattribute__(head)[ind]
                y = layer(conv)
                out[head] = y

            outs.append(out)
            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.convs_[ind](conv)
                inter = F.relu(inter)
                inter = self.inters[ind](inter)
        return outs


def make_hg_layers(kernel, dim0, dim1, mod, layer=conv_bn_relu):
    return Sequential(
        layer(kernel, dim0, dim1, stride=2),
        layer(kernel, dim1, dim1).repeat(mod-1),
    )


class HourglassNet(ExKp):
    def __init__(self, heads: Dict[str, int], num_stacks=2):
        n = 5
        dims = [256, 256, 384, 384, 384, 512]
        modules = [2, 2, 2, 2, 2, 4]

        super(HourglassNet, self).__init__(
            n, num_stacks, dims, modules, heads, conv_dim=256,
            make_hg_layers=make_hg_layers,
            kp_layer=Residual,
        )


if __name__ == '__main__':
    import numpy as np

    x = np.zeros((1, 3, 100, 100))
    r = Residual(3, 3, 10)

    print(r(x))
    assert r(x).shape == (1, 10, 100, 100)

    r = Residual(3, 3, 10, 2)
    assert r(x).shape == (1, 10, 50, 50)

    x = np.arange(25).reshape((1, 1, 5, 5)).astype(np.float32)
    print(x)
    print(F.unpooling_2d(x, 2, 2, cover_all=False))

    hg_net = HourglassNet({"hm": 3, "sizes": 1}, 2)

    x = np.zeros((1, 3, 512, 512)).astype(np.float32)

    print(hg_net(x))
