import os
import pickle

import chainer
from chainer.dataset import concat_examples
from chainer.functions.activation.sigmoid import sigmoid
from chainercv.datasets import voc_bbox_label_names, VOCBboxDataset
from chainercv.visualizations import vis_bbox

from centernet.datasets.transforms import CenterDetectionTransform
from centernet.models.center_detector import CenterDetector
from centernet.models.networks.hourglass import HourglassNet

import matplotlib.pyplot as plt
import cv2
import numpy as np


if __name__ == '__main__':
    image = cv2.imread("./data/demo/dog.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose((2, 0, 1))
    size = 256
    transform = CenterDetectionTransform(size, 100, 4)
    with chainer.using_config('train', False):
        num_class = len(voc_bbox_label_names)
        detector = CenterDetector(HourglassNet, size, num_class)
        chainer.serializers.load_npz('result/detector098.npz', detector)
        predicted = detector.predict([image])
        thresh_idx = predicted[2][0] > 0.3
        ax = vis_bbox(
            image,
            predicted[0][0][thresh_idx],
            predicted[1][0][thresh_idx],
            predicted[2][0][thresh_idx],
            label_names=voc_bbox_label_names,
        )
        plt.show()
