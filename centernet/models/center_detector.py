import chainer
import numpy as np
from chainer import Chain, Variable
from typing import Callable, Dict

from chainer.links import Classifier
from chainercv import transforms
from chainercv.links import SSD512

from centernet.functions.losses import center_detection_loss
from centernet.utilities import find_peak


class CenterDetector(Chain):

    def __init__(self, base_network_factory: Callable[[Dict[str, int]], Chain], insize, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.insize = insize
        with self.init_scope():
            self.base_network = base_network_factory({
                'hm': num_classes,
                'wh': 2,
                'offset': 2,
            })

    def forward(self, x):
        y = self.base_network(x)
        return y

    def predict(self, imgs, k=100):
        x = []
        sizes = []
        for img in imgs:
            _, H, W = img.shape
            img = self._prepare(img)
            x.append(self.xp.array(img))
            sizes.append((H, W))
        with chainer.using_config('train', False), chainer.function.no_backprop_mode():
            x = Variable(self.xp.stack(x))
            output = self.forward(x)[-1]

        bboxes = []
        labels = []
        scores = []
        for i in range(len(imgs)):
            bbox, label, score = self.decode_output(output, i, k)
            transforms.resize_bbox(bbox, (self.insize, self.insize), sizes[i])
            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)

        return bboxes, labels, scores

    def decode_output(self, output, index, k):
        bboxes = []
        labels = []
        scores = []
        for j in range(self.num_class):
            hm = output['hm'][index, j]
            hm.to_cpu()
            hm = hm.data  # type: np.ndarray

            indices = np.argsort(hm.flatten())[::-1][:k]
            for index in indices:
                x = index % hm.shape[1]
                y = index // hm.shape[1]
                peak_x, peak_y = find_peak(hm, x, y)

                x, y, w, h = self._decode_bbox(output, peak_x, peak_y, index)
                bboxes.append([x, y, x + w, y + h])
                labels.append(j)
                scores.append(hm[y, x])
        return np.array(bboxes), np.array(labels), np.array(scores)

    def _decode_bbox(self, output, x, y, index):
        wh = output['wh']
        wh.to_cpu()
        offset = output['offset']
        offset.to_cpu()

        return (
            x + offset[index, 0, y, x], y + offset[index, 1, y, x],
            wh[index, 0, y, x], wh[index, 1, y, x]
        )

    def _prepare(self, img):
        img = img.astype(np.float32)
        img = transforms.resize(img, (self.insize, self.insize))
        return img


class CenterDetectorTrain(Chain):
    def __init__(self, center_detector, hm_weight, wh_weight, offest_weight):
        super().__init__()

        self.hm_weight = hm_weight
        self.wh_weight = wh_weight
        self.offset_weight = offest_weight

        with self.init_scope():
            self.center_detector = center_detector

    def forward(self, **indata):
        imgs = indata['image']
        y = self.center_detector(imgs)
        loss = center_detection_loss(y, indata, self.hm_weight, self.wh_weight, self.offset_weight)
        return loss
