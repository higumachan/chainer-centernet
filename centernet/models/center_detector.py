from collections import defaultdict

import chainer
import numpy as np
from chainer import Chain, Variable, reporter
from typing import Callable, Dict

import chainer.functions as F
from chainer.links import Classifier
from chainercv import transforms
from chainercv.links import SSD512

from centernet.functions.losses import center_detection_loss
from centernet.utilities import find_peak


class CenterDetector(Chain):

    def __init__(self, base_network_factory: Callable[[Dict[str, int]], Chain], insize, num_classes, downratio=4):
        super().__init__()
        self.num_classes = num_classes
        self.insize = insize
        self.downratio = downratio
        with self.init_scope():
            self.base_network = base_network_factory({
                'hm': num_classes,
                'wh': 2,
                'offset': 2,
            })

    def forward(self, x):
        y = self.base_network(x)
        return y

    def predict(self, imgs, k=100, detail=False, output_index=-1):
        x = []
        sizes = []
        for img in imgs:
            _, H, W = img.shape
            img = self._prepare(img)
            x.append(self.xp.array(img))
            sizes.append((H, W))
        with chainer.using_config('train', False), chainer.function.no_backprop_mode():
            x = Variable(self.xp.stack(x))
            output = self.forward(x)[output_index]

        bboxes = []
        labels = []
        scores = []
        output['hm'] = F.sigmoid(output['hm'])
        output['hm'].to_cpu()
        for i in range(len(imgs)):
            bbox, label, score = self._decode_output(output, i, k)
            bbox = transforms.resize_bbox(bbox, (self.insize, self.insize), sizes[i])
            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)

        if detail:
            return bboxes, labels, scores, output
        else:
            return bboxes, labels, scores

    def _decode_output(self, output, index, k):
        bboxes = []
        labels = []
        scores = []
        for j in range(self.num_classes):
            hm = output['hm'].array[index, j]

            pos_indices = np.argsort(hm.flatten())[::-1][:k]
            already_visit_peak = defaultdict(lambda: False)
            for pos_index in pos_indices:
                x = pos_index % hm.shape[1]
                y = pos_index // hm.shape[1]
                peak_x, peak_y = find_peak(hm, x, y)
                if not already_visit_peak[peak_y, peak_x]:
                    already_visit_peak[peak_y, peak_x] = True

                    adjusted_x, adjusted_y, w, h = self._decode_bbox(output, peak_x, peak_y, index)
                    bboxes.append([adjusted_y - h / 2, adjusted_x - w / 2, adjusted_y + h / 2, adjusted_x + w / 2])
                    labels.append(j)
                    scores.append(hm[y, x])
        scores = np.array(scores)
        sorted_idx = scores.argsort()[::-1]
        return np.array(bboxes)[sorted_idx], np.array(labels)[sorted_idx], scores[sorted_idx]

    def _decode_bbox(self, output, x, y, index):
        wh = output['wh']
        wh.to_cpu()
        offset = output['offset']
        offset.to_cpu()
        wh = wh.array
        offset = offset.array

        return (
            x * self.downratio + offset[index, 0, y, x], y * self.downratio + offset[index, 1, y, x],
            wh[index, 0, y, x] * self.downratio, wh[index, 1, y, x] * self.downratio
        )

    def _prepare(self, img):
        img = img.astype(np.float32)
        img = transforms.resize(img, (self.insize, self.insize))
        return img


class CenterDetectorTrain(Chain):
    def __init__(self,
                 center_detector,
                 hm_weight,
                 wh_weight,
                 offest_weight,
                 focial_loss_alpha=2,
                 focial_loss_beta=4,
                 comm=None
                 ):
        super().__init__()

        self.hm_weight = hm_weight
        self.wh_weight = wh_weight
        self.offset_weight = offest_weight
        self.comm = comm

        with self.init_scope():
            self.center_detector = center_detector

    def forward(self, **indata):
        imgs = indata['image']
        y = self.center_detector(imgs)
        loss, hm_loss, wh_loss, offset_loss, detail_losses = center_detection_loss(
            y, indata,
            self.hm_weight, self.wh_weight, self.offset_weight, comm=self.comm
        )
        hm = y[-1]["hm"]
        hm_mae = F.mean_absolute_error(hm, indata["hm"])
        reporter.report({
            'loss': loss,
            'hm_loss': hm_loss,
            'hm_pos_loss': detail_losses['hm_pos_loss'],
            'hm_neg_loss': detail_losses['hm_neg_loss'],
            'hm_mae': hm_mae,
            'wh_loss': wh_loss,
            'offset_loss': offset_loss
        }, self)
        return loss


if __name__ == '__main__':
    from centernet.datasets.transforms import CenterDetectionTransform
    from chainercv.datasets import VOCBboxDataset
    from chainer.datasets import TransformDataset
    from chainer.dataset import concat_examples
    from centernet.models.networks.hourglass import HourglassNet

    center_detection_transform = CenterDetectionTransform(512, 5, 4)

    train = VOCBboxDataset(year='2012', split='trainval')

    x = concat_examples([train[0]])

    print(x[0].shape)
    detector = CenterDetector(HourglassNet, 512, 5)
    print(detector.predict(x[0]))
