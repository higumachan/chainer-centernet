import numpy as np
import math

from chainer.dataset import DatasetMixin
from chainercv import transforms
from chainercv.datasets import VOCBboxDataset, COCOBboxDataset

from centernet.utilities import gaussian_radius, draw_umich_gaussian


class CenterDetectionTransform:
    def __init__(self, size, num_classes, downratio) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.downratio = downratio
        self.size = size

    def __call__(self, in_data):
        img, bboxes, labels = in_data[:3]

        _, H, W, = img.shape
        img = transforms.resize(img, (self.size, self.size))
        bboxes = transforms.resize_bbox(bboxes, (H, W), (self.size, self.size))

        output_h = img.shape[1] // self.downratio
        output_w = img.shape[2] // self.downratio
        hm = np.zeros((self.num_classes, output_h, output_w), dtype=np.float32)
        dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)
        dense_offset = np.zeros((2, output_h, output_w), dtype=np.float32)
        dense_mask = np.zeros((2, output_h, output_w), dtype=np.float32)

        for i, (bbox, label) in enumerate(zip(bboxes, labels)):
            w = bbox[3] - bbox[1]
            h = bbox[2] - bbox[0]

            radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            radius = max(0, int(radius))

            center = np.array([
                (bbox[3] + bbox[1]) / 2, (bbox[2] + bbox[0]) / 2
            ], dtype=np.float32)
            center_int = center.astype(np.int32)
            draw_umich_gaussian(hm[label], center_int, radius)
            dense_wh[0, center_int[1], center_int[0]] = w
            dense_wh[1, center_int[1], center_int[0]] = h
            dense_offset[0, center_int[1], center_int[0]] = (center - center_int)[0]
            dense_offset[1, center_int[1], center_int[0]] = (center - center_int)[1]
            dense_mask[0, center_int[1], center_int[0]] = 1.0
            dense_mask[0, center_int[1], center_int[0]] = 1.0

        return {
            'image': img,
            'hm': hm,
            'dense_wh': dense_wh, 'dense_mask': dense_mask, 'dense_offset': dense_offset
        }

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    dataset = COCOBboxDataset()
    print(dataset)

