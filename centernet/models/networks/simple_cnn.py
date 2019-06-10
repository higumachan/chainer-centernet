from chainer import Chain

from centernet.models.networks.utilities import conv_bn_relu


class SimpleCNN(Chain):

    def __init__(self, heads):
        super().__init__()

        with self.init_scope():
            self.cnn1 = conv_bn_relu(3, 3, 32, 2)
            self.cnn2 = conv_bn_relu(3, 32, 64, 2)
            self.cnn3 = conv_bn_relu(3, 64, 64, 1)
            self.cnn4 = conv_bn_relu(3, 64, 64, 1)

            for head in heads:
                self.__setattr__(head, conv_bn_relu(3, 64, heads[head]))
            self.heads = heads

    def forward(self, x):
        h = self.cnn1(x)
        h = self.cnn2(h)
        h = self.cnn3(h)
        h = self.cnn4(h)

        out = {}

        for head in self.heads:
            layer = self.__getattribute__(head)
            y = layer(h)
            out[head] = y

        return [out]

if __name__ == '__main__':
    import numpy as np

    x = np.arange(25).reshape((1, 1, 5, 5)).astype(np.float32)
    print(x)

    hg_net = SimpleCNN({"hm": 3, "sizes": 1})

    x = np.zeros((1, 3, 512, 512)).astype(np.float32)

    print(hg_net(x))
