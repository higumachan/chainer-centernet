
import time
import numpy as np
from chainercv.links import ResNet50, Conv2DBNActiv


def test_resnet():
    resnet = ResNet50()
    print(resnet.layer_names)
    def forward(self, x):
        h = self.conv(x)
        if self.activ is None:
            return h
        else:
            return self.activ(h)

    Conv2DBNActiv.forward = forward
    print(resnet['conv1'])
    print(resnet['res2'])
    print(resnet(np.zeros((10, 3, 244, 244), dtype=np.float32)))


if __name__ == '__main__':
    s = time.time()
    test_resnet()
    e = time.time()

    print(f"{e - s} sec")

