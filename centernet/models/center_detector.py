from chainer import Chain
from typing import Callable, Dict

from chainer.links import Classifier

from centernet.functions.losses import center_detection_loss


class CenterDetector(Chain):

    def __init__(self, base_network_factory: Callable[[Dict[str, int]], Chain], num_classes):
        super().__init__()
        with self.init_scope():
            self.base_network = base_network_factory({
                'hm': num_classes,
                'wh': 2,
                'offset': 2,
            })

    def forward(self, x):
        y = self.base_network(x)
        return y[-1]


class CenterDetectorTrain(Chain):
    def __init__(self, base_network, hm_weight, wh_weight, offest_weight):
        super().__init__()

        self.hm_weight = hm_weight
        self.wh_weight = wh_weight
        self.offset_weight = offest_weight

        with self.init_scope():
            self.base_network = base_network

    def forward(self, x, gts):
        y = self.base_network(x)
        loss = center_detection_loss(y, gts, self.hm_weight, self.wh_weight, self.offset_weight)
        return loss
